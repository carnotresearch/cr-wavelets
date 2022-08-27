# Copyright 2022 CR-Suite Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import numpy as np
from jax import jit, lax
import jax.numpy as jnp

import cr.nimble as crn
import cr.wavelets as crwt
from cr.nimble import promote_arg_dtypes
import cr.nimble.dsp as crdsp

from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from cr.nimble.compression import *


def decompose(signal, wavelet, level):
    wavelet = crwt.to_wavelet(wavelet)
    coeffs = crwt.wavedec(signal, wavelet, level=level)
    return coeffs

def reconstruct(coeffs, wavelet, level):
    signal = crwt.waverec(coeffs, wavelet)
    return signal


def threshold(coeffs, energy_fractions):
    """Thresholds multi-level wavelet decomposition coefficients by fraction of energy
    """
    out_coeffs = []
    out_binmaps = []
    for coef, frac in zip(coeffs, energy_fractions):
        coef, binmap = crdsp.energy_threshold(coef, frac)
        out_coeffs.append(coef)
        out_binmaps.append(binmap)
    return out_coeffs, out_binmaps


def scale_to_0_1(coeffs):
    """Scales multi-level wavelet decomposition coefficients to [0,1] range
    """
    out_coeffs = []
    out_shifts = []
    out_scales = []
    for coef in coeffs:
        coef, shift, scale = crdsp.scale_to_0_1(coef)
        out_coeffs.append(coef)
        out_shifts.append(shift)
        out_scales.append(scale)
    return out_coeffs, out_shifts, out_scales


def descale_from_0_1(coeffs, shifts, scales):
    """Descales multi-level wavelet decomposition coefficients from [0,1] range
    """
    out_coeffs = []
    for coef, shift, scale in zip(coeffs, shifts, scales):
        coef = crdsp.descale_from_0_1(coef, shift, scale)
        out_coeffs.append(coef)
    return out_coeffs


def quantize_1(scaled_coeffs, bits):
    """Quantizes multi-level wavelet decomposition coefficients in [0,1] range to bits per sample
    """
    quantized_coeffs = [crdsp.quantize_1(coef, bits) for coef in scaled_coeffs]
    return quantized_coeffs

def inv_quantize_1(quantized_coeffs, bits):
    """Inverse quantizes multi-level wavelet decomposition coefficients in [0,1] range from bits per sample
    """
    coeffs = [crdsp.inv_quantize_1(coef, bits) for coef in quantized_coeffs]
    return coeffs


def quantize_to_prd_target(signal, wavelet, level, 
    scaled_coeffs, shifts, scales, binmaps, max_prd):
    """Quantizes multi-level wavelet decomposition coefficients in [0,1] range
    to a given target percentage root mean square difference
    """
    num_bits = 9
    cur_prd = 0
    prd_vals = np.empty(num_bits)
    while num_bits >= 5 and cur_prd <= max_prd:
        num_bits -= 1
        quantized_coeffs = quantize_1(scaled_coeffs, num_bits)
        inv_quant_coeffs = inv_quantize_1(quantized_coeffs, num_bits)
        unscaled_coeffs = descale_from_0_1(inv_quant_coeffs, shifts, scales)
        reconstructed = reconstruct(add_zeros(unscaled_coeffs, binmaps), wavelet, level)
        cur_prd = crn.prd(signal, reconstructed)
        # print(f'bits: {num_bits}, prd: {cur_prd}')
        prd_vals[num_bits] = cur_prd
    if cur_prd > max_prd:
        num_bits += 1
        cur_prd = prd_vals[num_bits]
    quantized_coeffs = quantize_1(scaled_coeffs, num_bits)
    return quantized_coeffs, num_bits, cur_prd


def remove_zeros(coeffs, binmaps):
    """Removes zeros from thresholded multi-level wavelet decomposition coefficients
    """
    return [coef[binmap.astype(bool)] for coef, binmap in zip(coeffs, binmaps)]

def add_zeros(coeffs, binmaps):
    """Adds zeros back to nonzero thresholded multi-level wavelet decomposition coefficients
    """
    result = []
    for coef, binmap in zip(coeffs, binmaps):
        out = jnp.zeros(binmap.shape)
        out = out.at[binmap.astype(bool)].set(coef)
        result.append(out)
    return result

def combine_arrays(arrays):
    """Combines multiple arrays (of binary maps or coefficients) into a single array
    """
    return jnp.concatenate(arrays)


def split_coefs_binmaps(wavelet, level, num_samples, combined_coeffs, combined_binmaps):
    """Splits the binary maps and coefficient arrays for different wavelet levels
    """
    sizes = crwt.dwt_coeff_lengths(num_samples, wavelet, level)
    # add the entry for the approximation coefficients
    sizes.append(sizes[-1])
    # reverse the sizes
    sizes = sizes[::-1]
    binmaps = []
    coeffs = []
    b_start = 0
    c_start = 0
    for size in sizes:
        binmap = combined_binmaps[b_start:b_start + size]
        binmaps.append(binmap)
        b_start += size
        # count the number of nonzero entries
        nnz = binmap.sum()
        coef = combined_coeffs[c_start:c_start + nnz]
        coeffs.append(coef)
        c_start += nnz
    return coeffs, binmaps


def encode_cbss_to_bits(coeffs, binmaps, shifts, scales, qbits):
    """Encodes multi-level wavelet decomposition quantized coefficients, their
    binary maps, scaling shifts and scales and number of quantization bits into
    a single bitarray
    """
    a = bitarray()
    a.extend(int2ba(qbits, 4))
    for value in shifts:
        a.extend(float_to_bitarray(value))
    for value in scales:
        a.extend(float_to_bitarray(value))
    binmap_code = encode_binary_arr(binmaps)
    n = len(binmap_code)
    a.extend(int_to_bitarray(n))
    a.extend(binmap_code)
    coef_code = encode_uint_arr_fl(coeffs, qbits)
    a.extend(coef_code)
    return a


def decode_cbss_from_bits(wavelet, level, num_samples, bits: bitarray):
    """Decodes multi-level wavelet decomposition quantized coefficients, their
    binary maps, scaling shifts and scales and number of quantization bits from
    a bitarray
    """
    shifts = []
    scales = []
    qbits = ba2int(bits[:4])
    pos = 4
    for i in range(level+1):
        value, pos = read_float_from_bitarray(bits, pos)
        shifts.append(value)
    for i in range(level+1):
        value, pos = read_float_from_bitarray(bits, pos)
        scales.append(value)
    n, pos = read_int_from_bitarray(bits, pos)
    e = pos + n
    binmap_bits = bits[pos:e]
    pos = e
    binmaps = decode_binary_arr(binmap_bits)
    # number of coefficients
    n_coeffs = np.sum(binmaps)
    n_coefbits = n_coeffs * qbits
    e = pos + n_coefbits
    coef_bits = bits[pos:e]
    pos = e
    coeffs = decode_uint_arr_fl(coef_bits, qbits)
    return coeffs, binmaps, shifts, scales, qbits
