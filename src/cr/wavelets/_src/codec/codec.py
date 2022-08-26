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

import cr.wavelets as crwt
from cr.nimble import promote_arg_dtypes
import cr.nimble.dsp as crdsp


def decompose(signal, wavelet, level):
    wavelet = crwt.to_wavelet(wavelet)
    coeffs = crwt.wavedec(signal, wavelet, level=level)
    return coeffs

def reconstruct(coefs, wavelet, level):
    signal = crwt.waverec(coeffs, wavelet)


def threshold(coeffs, energy_fractions):
    out_coeffs = []
    out_binmaps = []
    for coef, frac in zip(coeffs, energy_fractions):
        binmap, coef = crdsp.energy_threshold(tmp_coeffs, frac)
        out_binmaps.append(binmap)
        out_coeffs.append(coef)
    return out_coeffs, out_binmaps


def scale_to_0_1(coeffs):
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
    out_coeffs = []
    for coef, shift, scale in zip(coeffs, shifts, scales):
        coef = crdsp.descale_from_0_1(coef, shift, scale)
        out_coeffs.append(coef)
    return out_coeffs


def quantize_1(scaled_coeffs, bits):
    quantized_coeffs = []
    for coef in coeffs:
        coef = crdsp.quantize_1(scaled_coeffs, bits)
        quantized_coeffs.append(coef)
    return quantized_coeffs


def quantize_to_prd_target(signal, scaled_coeffs, shifts, scales, max_prd):
    num_bits = 9
    cur_prd = 0
    prd_vals = np.empty(num_bits)
    while num_bits >= 5 and cur_prd <= max_prd:
        num_bits -= 1
        quantized_coeffs = quantize_1(scaled_coeffs, num_bits)
        inv_quant_coeffs = inv_quantize_1(quantized_coeffs, num_bits)
        unscaled_coeffs = descale_from_0_1(inv_quant_coeffs, shifts, scales)
        reconstructed = reconstruct(unscaled_coeffs)
        cur_prd = crn.prd(signal, reconstructed)
        prd_vals[num_bits] = cur_prd
    if cur_prd > max_prd:
        num_bits += num_bits
        cur_prd = prd_vals[num_bits]
    quantized_coeffs = quantize_1(scaled_coeffs, num_bits)
    return quantized_coeffs, num_bits, cur_prd

def combine_coefficients(quantized_coeffs, binmaps):
    return jnp.concatenate((coef[binmap] 
        for coef, binmap in zip(quantized_coeffs, binmaps)))

def combine_binary_maps(binmaps):
    return jnp.concatenate(binmaps)
