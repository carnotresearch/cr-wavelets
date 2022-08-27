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

import cr.wavelets as crwt
from .blocks import *

def build_codec(wavelet_name, level,
    block_length, max_prd, thresholds):
    """Builds an encoder and decoder function for a given choice
    of wavelet, decomposition level, signal block length, target
    percentage root mean square difference and coefficient energy thresholding
    targets
    """
    wavelet = crwt.to_wavelet(wavelet_name)


    def encode(block):
        coeffs = crwt.wavedec(block, wavelet, level=level)
        th_coeffs, binmaps = threshold(coeffs, thresholds)
        nz_th_coeffs = remove_zeros(th_coeffs, binmaps)
        scaled_coeffs, shifts, scales = scale_to_0_1(nz_th_coeffs)
        # quantize_to_prd_target
        num_bits = 9
        cur_prd = 0
        prd_vals = np.empty(num_bits)
        while num_bits >= 5 and cur_prd <= max_prd:
            num_bits -= 1
            quantized_coeffs = quantize_1(scaled_coeffs, num_bits)
            inv_quant_coeffs = inv_quantize_1(quantized_coeffs, num_bits)
            unscaled_coeffs = descale_from_0_1(inv_quant_coeffs, shifts, scales)
            unscaled_coeffs = add_zeros(unscaled_coeffs, binmaps)
            reconstructed = crwt.waverec(unscaled_coeffs, wavelet)
            cur_prd = crn.prd(block, reconstructed)
            prd_vals[num_bits] = cur_prd
        if cur_prd > max_prd:
            num_bits += 1
            cur_prd = prd_vals[num_bits]
        quantized_coeffs = quantize_1(scaled_coeffs, num_bits)
        combined_coeffs = combine_arrays(quantized_coeffs)
        combined_binmaps = combine_arrays(binmaps)
        result = encode_cbss_to_bits(
            combined_coeffs, combined_binmaps, 
            shifts, scales, num_bits)
        return result

    def decode(bits):
        (c_coeffs, c_binmaps, 
            shifts, scales, num_bits) = decode_cbss_from_bits(
            wavelet_name, level, block_length, bits)
        coeffs, binmaps = split_coefs_binmaps(
            wavelet_name, level, block_length, c_coeffs, c_binmaps)
        inv_quant_coeffs = inv_quantize_1(coeffs, num_bits)
        unscaled_coeffs = descale_from_0_1(inv_quant_coeffs, shifts, scales)
        unscaled_coeffs = add_zeros(unscaled_coeffs, binmaps)
        reconstructed = crwt.waverec(unscaled_coeffs, wavelet)
        return reconstructed

    return encode, decode


