# Copyright 2021 CR-Suite Development Team
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


from enum import Enum, auto
from typing import NamedTuple, List, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft

from .families import FAMILY, wname_to_family_order, is_discrete_wavelet
from .coeffs import db, sym, coif, bior, dmey, sqrt2

from .cont_wavelets import WaveletFunctions, cmor, ricker

import re


class SYMMETRY(Enum):
    """Describes the type of symmetry in a wavelet
    """
    UNKNOWN = -1
    """Unknown Symmetry"""
    ASYMMETRIC = 0
    """Assymetric Wavelet"""
    NEAR_SYMMETRIC = 1
    """Near Symmetric Wavelet"""
    SYMMETRIC = 2
    """Symmetric Wavelet"""
    ANTI_SYMMETRIC = 3
    """Anti-symmetric Wavelet"""



class BaseWavelet(NamedTuple):
    """Represents basic information about a wavelet
    """
    support_width: int = 0
    symmetry: SYMMETRY = SYMMETRY.UNKNOWN
    orthogonal: bool = False
    biorthogonal: bool = False
    compact_support: bool = False
    name: FAMILY = None
    family_name: str = None
    short_name: str = None

class DiscreteWavelet(NamedTuple):
    """Represents information about a discrete wavelet
    """
    support_width: int = -1
    """Length of the support for finite support wavelets"""
    symmetry: SYMMETRY = SYMMETRY.UNKNOWN
    """Indicates the kind of symmetry inside the wavelet"""
    orthogonal: bool = False
    """Indicates if the wavelet is orthogonal"""
    biorthogonal: bool = False
    """Indicates if the wavelet is biorthogonal"""
    compact_support: bool = False
    """Indicates if the wavelet has compact support"""
    name: str = ''
    """Name of the wavelet"""
    family_name: str = ''
    """Name of the wavelet family"""
    short_name: str = ''
    """Short name of the wavelet family"""
    dec_hi: jax.Array = None
    """Decomposition high pass filter"""
    dec_lo: jax.Array = None
    """Decomposition low pass filter"""
    rec_hi: jax.Array = None
    """Reconstruction high pass filter"""
    rec_lo: jax.Array = None
    """Reconstruction low pass filter"""
    dec_len: int = 0
    """Length of decomposition filters"""
    rec_len: int = 0
    """Length of reconstruction filters"""
    vanishing_moments_psi: int = 0
    """Number of vanishing moments of the wavelet function"""
    vanishing_moments_phi: int = 0
    """Number of vanishing moments of the scaling function"""

    def __str__(self):
        """Returns the string representation
        """
        s = []
        for x in [
            u"Wavelet %s"           % self.name,
            u"  Family name:    %s" % self.family_name,
            u"  Short name:     %s" % self.short_name,
            u"  Filters length: %d" % self.dec_len,
            u"  Orthogonal:     %s" % self.orthogonal,
            u"  Biorthogonal:   %s" % self.biorthogonal,
            u"  Symmetry:       %s" % self.symmetry.name.lower(),
            u"  DWT:            True",
            u"  CWT:            False"
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)

    def wavefun(self, level=8):
        """Returns the scaling and wavelet functions for the wavelet

        Args:
            level (:obj:`int`, optional): Number of levels of reconstruction 
                to get the approximation of scaling and wavelet functions. 
                Default 8.
        """
        from .discrete import orth_wavefun_jit, biorth_wavefun
        if self.orthogonal:
            return orth_wavefun_jit(self.rec_lo, self.rec_hi, level=level)
        if self.biorthogonal:
            return biorth_wavefun(self, level=level)
        raise NotImplementedError()

    @property
    def filter_bank(self):
        """Returns the Quadratrure Mirror Filter Bank associated with the wavelet (dec_lo, dec_hi, rec_lo, rec_hi)
        """
        return (self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi)

    @property
    def inverse_filter_bank(self):
        """Returns the filter bank associated with the inverse wavelet
        """
        return (self.rec_lo[::-1], self.rec_hi[::-1], 
            self.dec_lo[::-1], self.dec_hi[::-1])

class ContinuousWavelet(NamedTuple):
    """Represents information about a continuous wavelet
    """
    support_width: int = -1
    """Length of the support for finite support wavelets"""
    symmetry: SYMMETRY = SYMMETRY.UNKNOWN
    """Indicates the kind of symmetry inside the wavelet"""
    orthogonal: bool = False
    """Indicates if the wavelet is orthogonal"""
    biorthogonal: bool = False
    """Indicates if the wavelet is biorthogonal"""
    compact_support: bool = False
    """Indicates if the wavelet has compact support"""
    name: str = ''
    """Name of the wavelet"""
    family_name: str = ''
    """Name of the wavelet family"""
    short_name: str = ''
    """Short name of the wavelet family"""

    # additinal parameters for continuous wavelets
    lower_bound: float = 0
    """time window lower bound for computing the wavelet function"""
    upper_bound: float = 0
    """time window upper bound for computing the wavelet function"""
    complex_cwt: bool = False
    """flag indicating if the wavelet is complex or real"""
    center_frequency: float = -1.
    """center frequency of the wavelet"""
    bandwidth_frequency: float = -1.
    """bandwidth of the wavelet"""
    fbsp_order: int = 0
    functions: WaveletFunctions = None
    """Functions associated with the wavelet"""


    def __str__(self):
        s = []
        for x in [
            u"ContinuousWavelet  %s"           % self.name,
            u"  Family name:    %s" % self.family_name,
            u"  Short name:     %s" % self.short_name,
            u"  Symmetry:       %s" % self.symmetry.name.lower(),
            u"  DWT:            False",
            u"  CWT:            True",
            u"  Complex CWT:   %s" % self.complex_cwt,
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)

    def wavefun(self, level=8, length=None):
        """Returns the wavelet function for the wavelet

        Args:
            level (:obj:`int`, optional): Number of levels of reconstruction 
                to get the approximation of the wavelet function. 
                Default 8.
        """
        if self.functions is None:
            raise NotImplementedError(f"No implementation available for {self.name}")
        func = self.functions.time
        p = 2**level
        output_length = p if length is None else length
        t = jnp.linspace(self.lower_bound, self.upper_bound, output_length)
        psi = func(t)
        return psi, t

    @property
    def domain(self):
        """Returns the time domain of the wavelet
        """
        return self.upper_bound - self.lower_bound


def qmf(h):
    """Returns the quadrature mirror filter of a given filter"""
    g = h[::-1]
    g = g.at[1::2].set(-g[1::2])
    return g

def orthogonal_filter_bank(scaling_filter):
    """Returns the orthogonal filter bank for a given scaling filter"""
    # scaling filter must be even
    if not (scaling_filter.shape[0] % 2) == 0:
        raise ValueError('scaling_filter must be of even length.')
    # normalize
    rec_lo = sqrt2 * scaling_filter / jnp.sum(scaling_filter)
    dec_lo = rec_lo[::-1]
    rec_hi = qmf(rec_lo)
    dec_hi = rec_hi[::-1]
    return (dec_lo, dec_hi, rec_lo, rec_hi)

def filter_bank_(rec_lo):
    """Construct a filter bank from the saved values in coeffs.py"""
    dec_lo = rec_lo[::-1]
    rec_hi = qmf(rec_lo)
    dec_hi = rec_hi[::-1]
    return (dec_lo, dec_hi, rec_lo, rec_hi)

def mirror(h):
    n = h.shape[0]
    modulation = (-1)**jnp.arange(1, n+1)
    return modulation * h

def negate_evens(g):
    return g.at[0::2].set(-g[0::2])

def negate_odds(g):
    return g.at[1::2].set(-g[1::2])


def bior_index(n, m):
    idx = max = None
    if n == 1:
        idx = m // 2
        max = 5
    elif n == 2:
        idx = m // 2 -1 
        max = 8
    elif n == 3:
        idx = m // 2
        max = 9
    elif n == 4 or n == 5:
        if n == m:
            idx = 0
            max = m
    elif n == 6:
        if m == 8:
            idx = 0
            max = 8
    else:
        pass
    return idx, max


def build_discrete_wavelet(family: FAMILY, order: int):
    """Builds a descrete wavelet by its family and order
    """
    nv = family.value
    if nv is FAMILY.HAAR.value:
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank_(db[0])
        w = DiscreteWavelet(support_width=1,
            symmetry=SYMMETRY.ASYMMETRIC,
            orthogonal=True,
            biorthogonal=True,
            compact_support=True,
            name="Haar",
            family_name = "Haar",
            short_name="haar", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=2,
            rec_len=2,
            vanishing_moments_psi=1,
            vanishing_moments_phi=0)
        return w
    if nv == FAMILY.DB.value:
        index = order - 1
        if index >= len(db):
            return None
        filters_length = 2 * order
        dec_len = rec_len = filters_length
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank_(db[index])
        w = DiscreteWavelet(support_width=2*order-1,
            symmetry=SYMMETRY.ASYMMETRIC,
            orthogonal=True,
            biorthogonal=True,
            compact_support=True,
            name=f'db{order}',
            family_name = "Daubechies",
            short_name="db", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=dec_len,
            rec_len=rec_len,
            vanishing_moments_psi=order,
            vanishing_moments_phi=0)
        return w
    if nv == FAMILY.SYM.value:
        index = order - 2
        if index >= len(sym):
            return None
        filters_length = 2 * order
        dec_len = rec_len = filters_length
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank_(sym[index])
        w = DiscreteWavelet(support_width=2*order-1,
            symmetry=SYMMETRY.NEAR_SYMMETRIC,
            orthogonal=True,
            biorthogonal=True,
            compact_support=True,
            name=f'sym{order}',
            family_name = "Symlets",
            short_name="sym", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=dec_len,
            rec_len=rec_len,
            vanishing_moments_psi=order,
            vanishing_moments_phi=0)
        return w
    if nv == FAMILY.COIF.value:
        index = order - 1
        if index >= len(coif):
            return None
        filters_length = 6 * order
        dec_len = rec_len = filters_length
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank_(coif[index]*sqrt2)
        w = DiscreteWavelet(support_width=6*order-1,
            symmetry=SYMMETRY.NEAR_SYMMETRIC,
            orthogonal=True,
            biorthogonal=True,
            compact_support=True,
            name=f'coif{order}',
            family_name = "Coiflets",
            short_name="coif", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=dec_len,
            rec_len=rec_len,
            vanishing_moments_psi=2*order,
            vanishing_moments_phi=2*order-1)
        return w
    if nv == FAMILY.BIOR.value:
        n = order // 10
        m = order % 10
        idx, max = bior_index(n, m)
        if idx is None or max is None:
            return None
        arr = bior[n-1]
        if idx >= len(arr):
            return None
        filters_length = 2*m if n == 1 else 2*m + 2
        dec_len = rec_len = filters_length
        start = max - m
        rec_lo = arr[0][start:start+rec_len]
        dec_lo = arr[idx+1][::-1]
        rec_hi = negate_odds(dec_lo)
        dec_hi = negate_evens(rec_lo)
        w = DiscreteWavelet(support_width=6*order-1,
            symmetry=SYMMETRY.SYMMETRIC,
            orthogonal=False,
            biorthogonal=True,
            compact_support=True,
            name=f'bior{n}.{m}',
            family_name = "Biorthogonal",
            short_name="bior", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=dec_len,
            rec_len=rec_len,
            vanishing_moments_psi=2*order,
            vanishing_moments_phi=2*order-1)
        return w
    if nv == FAMILY.RBIO.value:
        n = order // 10
        m = order % 10
        idx, max = bior_index(n, m)
        if idx is None or max is None:
            return None
        arr = bior[n-1]
        if idx >= len(arr):
            return None
        filters_length = 2*m if n == 1 else 2*m + 2
        dec_len = rec_len = filters_length
        start = max - m
        dec_lo = arr[0][start:start+rec_len][::-1]
        rec_lo = arr[idx+1]
        rec_hi = negate_odds(dec_lo)
        dec_hi = negate_evens(rec_lo)
        w = DiscreteWavelet(support_width=6*order-1,
            symmetry=SYMMETRY.SYMMETRIC,
            orthogonal=False,
            biorthogonal=True,
            compact_support=True,
            name=f'rbio{n}.{m}',
            family_name = "Reverse biorthogonal",
            short_name="rbio", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=dec_len,
            rec_len=rec_len,
            vanishing_moments_psi=2*order,
            vanishing_moments_phi=2*order-1)
        return w
    if nv is FAMILY.DMEY.value:
        dec_len = rec_len = filters_length = 62
        dec_lo, dec_hi, rec_lo, rec_hi = filter_bank_(dmey)
        w = DiscreteWavelet(support_width=1,
            symmetry=SYMMETRY.SYMMETRIC,
            orthogonal=True,
            biorthogonal=True,
            compact_support=True,
            name="dmey",
            family_name = "Discrete Meyer (FIR Approximation)",
            short_name="dmey", 
            dec_hi=dec_hi,
            dec_lo=dec_lo,
            rec_hi=rec_hi,
            rec_lo=rec_lo,
            dec_len=dec_len,
            rec_len=rec_len,
            vanishing_moments_psi=-1,
            vanishing_moments_phi=-1)
        return w
    return None


# regular expression for finding bandwidth-frequency and center-frequency 
cwt_pattern = re.compile(r'\D+(\d+\.*\d*)+')


def _get_bw_center_freqs(freqs, bandwidth_frequency, center_frequency):
    if len(freqs) == 2:
        bandwidth_frequency = float(freqs[0])
        center_frequency = float(freqs[1])
    return  bandwidth_frequency, center_frequency

def _get_m_b_c(freqs, fbsp_order, bandwidth_frequency, center_frequency):
    if len(freqs) == 3:
        fbsp_order = int(freqs[0])
        bandwidth_frequency = float(freqs[1])
        center_frequency = float(freqs[2])
    return fbsp_order, bandwidth_frequency, center_frequency

def build_continuous_wavelet(name: str, family: FAMILY, order: int):
    """Builds a continuous wavelet by its family and order
    """
    # wavelet base name
    base_name = name[:4]
    subname = name[4:]
    # indentify the B-C pattern if present
    freqs = re.findall(cwt_pattern, name)
    if subname and len(freqs) == 0:
        raise ValueError("No frequencies have been specified.")
    freqs = [float(freq) for freq in freqs]
    nv = family.value
    if nv == FAMILY.GAUS.value:
        if order > 8:
            return None
        symmetry = SYMMETRY.SYMMETRIC if order % 2 == 0 else SYMMETRY.ANTI_SYMMETRIC
        w = ContinuousWavelet(support_width=-1,
            symmetry=symmetry,
            orthogonal=False,
            biorthogonal=False,
            compact_support=False,
            name=name,
            family_name = "Gaussian",
            short_name="gaus", 
            complex_cwt=False,
            lower_bound=-5.,
            upper_bound=5.,
            center_frequency=0.,
            bandwidth_frequency=0.,
            fbsp_order=0)
        return w
    elif nv == FAMILY.MEXH.value:
        functions = ricker()
        w = ContinuousWavelet(support_width=-1,
            symmetry=SYMMETRY.SYMMETRIC,
            orthogonal=False,
            biorthogonal=False,
            compact_support=False,
            name=name,
            family_name = "Mexican hat wavelet",
            short_name="mexh", 
            complex_cwt=False,
            lower_bound=-8.,
            upper_bound=8.,
            center_frequency=0.25,
            bandwidth_frequency=0.,
            fbsp_order=0,
            functions=functions)
        return w
    elif nv == FAMILY.MORL.value:
        w = ContinuousWavelet(support_width=-1,
            symmetry=SYMMETRY.SYMMETRIC,
            orthogonal=False,
            biorthogonal=False,
            compact_support=False,
            name=name,
            family_name = "Morlet wavelet",
            short_name="morl", 
            complex_cwt=False,
            lower_bound=-8.,
            upper_bound=8.,
            center_frequency=0.,
            bandwidth_frequency=0.,
            fbsp_order=0)
        return w
    elif nv == FAMILY.CGAU.value:
        if order > 8:
            return None
        symmetry = SYMMETRY.SYMMETRIC if order % 2 == 0 else SYMMETRY.ANTI_SYMMETRIC
        w = ContinuousWavelet(support_width=-1,
            symmetry=symmetry,
            orthogonal=False,
            biorthogonal=False,
            compact_support=False,
            name=name,
            family_name = "Complex Gaussian wavelets",
            short_name="cgau", 
            complex_cwt=True,
            lower_bound=-5.,
            upper_bound=5.,
            center_frequency=0.,
            bandwidth_frequency=0.,
            fbsp_order=0)
        return w
    elif nv == FAMILY.SHAN.value:
        bandwidth_frequency, center_frequency = _get_bw_center_freqs(freqs, 0.5, 1.)
        w = ContinuousWavelet(support_width=-1,
            symmetry=SYMMETRY.ASYMMETRIC,
            orthogonal=False,
            biorthogonal=False,
            compact_support=False,
            name=name,
            family_name = "Shannon wavelets",
            short_name="shan", 
            complex_cwt=True,
            lower_bound=-20.,
            upper_bound=20.,
            center_frequency=center_frequency,
            bandwidth_frequency=bandwidth_frequency,
            fbsp_order=0)
        return w
    elif nv == FAMILY.FBSP.value:
        fbsp_order, bandwidth_frequency, center_frequency = _get_m_b_c(freqs, 2, 1., 0.5)
        w = ContinuousWavelet(support_width=-1,
            symmetry=SYMMETRY.ASYMMETRIC,
            orthogonal=False,
            biorthogonal=False,
            compact_support=False,
            name=name,
            family_name = "Frequency B-Spline wavelets",
            short_name="fbsp", 
            complex_cwt=True,
            lower_bound=-20.,
            upper_bound=20.,
            center_frequency=center_frequency,
            bandwidth_frequency=bandwidth_frequency,
            fbsp_order=fbsp_order)
        return w
    elif nv == FAMILY.CMOR.value:
        bandwidth_frequency, center_frequency = _get_bw_center_freqs(freqs, 1., 0.5)
        functions = cmor(bandwidth_frequency, center_frequency)
        w = ContinuousWavelet(support_width=-1,
            symmetry=SYMMETRY.ASYMMETRIC,
            orthogonal=False,
            biorthogonal=False,
            compact_support=False,
            name=name,
            family_name = "Complex Morlet wavelets",
            short_name="cmor", 
            complex_cwt=True,
            lower_bound=-8.,
            upper_bound=8.,
            center_frequency=center_frequency,
            bandwidth_frequency=bandwidth_frequency,
            fbsp_order=2,
            functions=functions)
        return w
    return None

def build_wavelet(name):
    """Builds a wavelet object by the name of the wavelet

    Args:
        name (str): Name of the wavelet

    Returns:
        cr.sparse.wt.DiscreteWavelet: a discrete wavelet object

    Example:
        ::

            >>> wavelet = wt.build_wavelet('db1')
            >>> print(wavelet)
            Wavelet db1
                Family name:    Daubechies
                Short name:     db
                Filters length: 2
                Orthogonal:     True
                Biorthogonal:   True
                Symmetry:       asymmetric
                DWT:            True
                CWT:            False
            >>> dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
            >>> print(dec_lo)
            >>> print(dec_hi)
            >>> print(rec_lo)
            >>> print(rec_hi)
            [0.70710678 0.70710678]
            [-0.70710678  0.70710678]
            [0.70710678 0.70710678]
            [ 0.70710678 -0.70710678]
            >>> phi, psi, x = wavelet.wavefun()
    """
    name = name.lower()
    family, order = wname_to_family_order(name)
    wavelet = None
    if is_discrete_wavelet(family):
        wavelet = build_discrete_wavelet(family, order)
    else:
        wavelet = build_continuous_wavelet(name, family, order)
    # other wavelet types are not supported for now
    if wavelet is None:
        raise ValueError(f"Invalid wavelet name {name}")
    return wavelet

def rec_integrate(function, dt):
    """Integrate a function using the rectangle integration method
    """
    integral = jnp.cumsum(function)
    integral *= dt
    return integral


def to_wavelet(wavelet):
    if isinstance(wavelet, str):
        wavelet = build_wavelet(wavelet)
    if wavelet is None:
        raise ValueError("Invalid wavelet")
    return wavelet

def integrate_wavelet(wavelet, precision=8):
    """Integrate wavelet function using the rectangle integration method
    """
    wavelet = to_wavelet(wavelet)
    approximations = wavelet.wavefun(precision)
    if len(approximations) == 2:
        psi, t = approximations
        dt = t[1] - t[0]
        return rec_integrate(psi, dt), t
    elif len(approximations) == 3:
        phi, psi, t = approximations
        dt = t[1] - t[0]
        return rec_integrate(psi, dt), t
    elif len(approximations) == 5:
        phi_d, psi_d, phi_r, psi_r, t = approximations
        dt = t[1] - t[0]
        return rec_integrate(psi_d, dt), rec_integrate(psi_r, dt), t


def central_frequency(wavelet, precision=8):
    """Computes the central frequency of the wavelet function
    """
    wavelet = to_wavelet(wavelet)
    # Let's see if the central frequency is defined for the wavelet
    if wavelet.center_frequency:
        return wavelet.center_frequency
    # get the wavelet functions
    approximations = wavelet.wavefun(precision)
    if len(approximations) == 2:
        psi, t = approximations
    elif len(approximations) == 3:
        _, psi, t = approximations
    elif len(approximations) == 5:
        _, psi, _, _, t = functions_approximations
    domain = t[-1] - t[0]
    # identify the peak frequency [skip the DC]
    index = jnp.argmax(jnp.abs(jfft.fft(psi)[1:])) + 2
    if index > len(psi) / 2:
        index = len(psi) - index + 2
    # convert to Hz
    return 1.0 / (domain / (index - 1))
    
    
def scale2frequency(wavelet, scales, precision=8):
    """Converts scales to frequencies for a wavelet
    """
    scales = jnp.asarray(scales)
    cf  = central_frequency(wavelet, precision=precision)
    return cf / scales
