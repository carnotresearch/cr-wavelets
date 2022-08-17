"""
API similar to WaveLab
"""
# pylint: disable=W0611


######################################################################################
# Wavelab based implementation
######################################################################################

from cr.wavelets._src.conv import (
    iconv,
    aconv,
    mirror_filter,
)

from cr.wavelets._src.multirate import (
    up_sample,
    lo_pass_down_sample,
    hi_pass_down_sample,
    up_sample_lo_pass,
    up_sample_hi_pass,
    downsampling_convolution_periodization,
)


from cr.wavelets._src.orth import (
    wavelet_function,
    scaling_function,
    haar,
    db4,
    db6,
    db8,
    db10,
    db12,
    db14,
    db16,
    db18,
    db20,
    baylkin,
    coif1,
    coif2,
    coif3,
    coif4,
    coif5,
    symm4,
    symm5,
    symm6,
    symm7,
    symm8,
    symm9,
    symm10,
    vaidyanathan,
)

from cr.wavelets._src.transform import (
    ## multi-level transforms
    forward_periodized_orthogonal,
    forward_periodized_orthogonal_jit,
    inverse_periodized_orthogonal,
    inverse_periodized_orthogonal_jit
)

