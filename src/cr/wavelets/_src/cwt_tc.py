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

import math
from functools import partial
from typing import NamedTuple, Callable, Tuple

from jax import jit, lax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import scipy
import numpy as np
import matplotlib.pyplot as plt

from .cont_wavelets import *

from .util import next_pow_of_2, time_points, frequency_points
from .wavelet import to_wavelet

########################################################################################################
# CWT in time and frequency domains
########################################################################################################

def cwt_tc_time(data, wavelet_func, scales, dt=1., axis=-1):
    """Computes the continuous wavelet transform
    """
    sample = wavelet_func(1, 1.)
    a = len(scales)
    n = data.shape[axis]
    out_shape = (a,) + data.shape
    output = jnp.empty(out_shape, dtype=sample.dtype)
    # compute in time
    slices = [None for _ in data.shape]
    slices[axis] = slice(None)
    slices = tuple(slices)
    t = time_points(n, dt)
    for index, scale in enumerate(scales):
        # n = jnp.minimum(10*scale, b)
        # sample wavelet and normalise
        norm = (dt) ** .5
        # compute the wavelet
        wavelet_seq = norm * wavelet_func(t, scale)
        # keep a max of 10:scale values
        # wavelet = wavelet[:10*scale]
        # conjugate it
        wavelet_seq = jnp.conj(wavelet_seq)
        # reverse it
        wavelet_seq = wavelet_seq[::-1]
        filter = wavelet_seq[slices]
        if jnp.isrealobj(filter):
            # convolve with data
            coeffs = jnp.convolve(data, filter, mode='same')
        else:
            # convolve with data
            coeffs_real = jnp.convolve(data, filter.real, mode='same')
            coeffs_imag = jnp.convolve(data, filter.imag, mode='same')
            coeffs = lax.complex(coeffs_real, coeffs_imag)
        output = output.at[index].set(coeffs)
    return output


cwt_tc_time_jit = jit(cwt_tc_time, static_argnums=(1,3,4))



def cwt_tc_frequency(data, wavelet_func, scales, dt=1., axis=-1):
    """
    Computes the CWT of data [along axis] for a given wavelet (in frequency domain)
    """
    # make sure that parmeters are arrays
    data = jnp.asarray(data)
    scales = jnp.asarray(scales)
    # number of data points for each data vector
    n  = data.shape[axis]
    # next power of 2
    pn = next_pow_of_2(n)
    # compute the FFT of the data
    data_fft = jfft.fft(data, n=pn, axis=axis)
    # angular frequencies at which the Wavelet basis will be computed
    wk = jfft.fftfreq(pn, d=dt) * 2 * jnp.pi
    # sample wavelet at all the scales and normalise
    norm = ( 1 / dt) ** .5
    wavelet_freq = norm * wavelet_func(wk, scales)
    # take the conjugate
    wavelet_freq = jnp.conj(wavelet_freq)
    # Convert negative axis. Add one to account for
    # inclusion of scales axis above.
    axis = (axis % data.ndim) + 1
    # perform the convolution in frequency space
    slices = [slice(None)] + [None for _ in data.shape]
    slices[axis] = slice(None)
    slices = tuple(slices)
    out = jfft.ifft(data_fft[None] * wavelet_freq[slices],
                     n=pn, axis=axis)
    slices = [slice(None) for _ in out.shape]
    slices[axis] = slice(None, n)
    slices = tuple(slices)
    if data.ndim == 1:
        return out[slices].squeeze()
    else:
        return out[slices]

cwt_tc_frequency_jit = jit(cwt_tc_frequency, static_argnums=(1,3, 4))

def cwt_tc(data, scales, wavelet, sampling_period=1., method='conv', axis=-1):
    """Computes the CWT of data along a specified axis with a specified wavelet
    """
    wavelet = to_wavelet(wavelet)
    if method == 'conv':
        wavelet_func = wavelet.functions.time
        output = cwt_tc_time_jit(data, wavelet_func, scales, dt=sampling_period, axis=axis)
    elif method == 'fft':
        wavelet_func = wavelet.functions.frequency
        output = cwt_tc_frequency_jit(data, wavelet_func, scales, dt=sampling_period, axis=axis)
    else:
        raise NotImplementedError("The specified method is not supported yet")
    return output


########################################################################################################
# Tuple Describing a Continuous Wavelet Analysis Result
########################################################################################################

class WaveletAnalysis(NamedTuple):
    """Continuous Wavelet Analysis of a 1D data signal
    """
    data : jnp.ndarray
    """ data on which analysis is being performed"""
    wavelet: WaveletFunctions
    """ The wavelet being used for analysis"""
    dt: float
    """ sample spacing / period"""
    dj : float
    """ scale resolution """
    mask_coi: bool
    """Disregard wavelet power outside the cone of influence"""
    frequency: bool 
    """The method used for computing CWT time domain or frequency domain"""
    axis : int
    """ The axis along which the analysis will be performed"""
    scales: jnp.ndarray
    """ The scales at which the analysis was performed"""
    scalogram : jnp.ndarray
    """The resultant scalogram"""

    @property
    def n(self):
        """Returns the length of data along the axis on which CWT is being computed"""
        return self.data.shape[self.axis]

    @property
    def times(self):
        """Returns the nomal time points for the dataset"""
        return time_points(self.n, self.dt)

    @property
    def fourier_period(self):
        """Return a function that calculates the equivalent Fourier
        period as a function of scale.
        """
        return self.wavelet.fourier_period

    @property
    def scale_from_period(self):
        """Return a function that calculates the wavelet scale
        from the fourier period
        """
        return self.wavelet.scale_from_period

    @property
    def fourier_periods(self):
        """Return the equivalent Fourier periods for the scales used."""
        return self.fourier_period(self.scales)

    @property
    def fourier_frequencies(self):
        """
        Return the equivalent frequencies .
        This is equivalent to 1.0 / self.fourier_periods
        """
        return jnp.reciprocal(self.fourier_periods)

    @property
    def s0(self):
        return find_s0(self.wavelet, self.dt)

    @property
    def w_k(self):
        """Angular frequency as a function of Fourier index.
        N.B the frequencies returned by numpy are adimensional, on
        the interval [-1/2, 1/2], so we multiply by 2 * pi.
        """
        return 2 * jnp.pi * jfft.fftfreq(self.n, self.dt)

    @property
    def magnitude(self):
        """Returns the magnitude of the scalogram"""
        return jnp.abs(self.scalogram)

    @property
    def power(self):
        """Calculate the wavelet power spectrum"""
        return jnp.abs(self.scalogram) ** 2

    @property
    def coi(self):
        """The Cone of Influence is the region near the edges of the
        input signal in which edge effects may be important.

        Return a tuple (T, S) that describes the edge of the cone
        of influence as a single line in (time, scale).
        """
        times = self.times
        scales = self.scales
        Tmin = times.min()
        Tmax = times.max()
        Tmid = Tmin + (Tmax - Tmin) / 2
        s = np.logspace(np.log10(scales.min()),
                        np.log10(scales.max()),
                        100)
        coi_func = self.wavelet.coi
        c1 = Tmin + coi_func(s)
        c2 = Tmax - coi_func(s)

        C = np.hstack((c1[np.where(c1 < Tmid)], c2[np.where(c2 > Tmid)]))
        S = np.hstack((s[np.where(c1 < Tmid)], s[np.where(c2 > Tmid)]))

        # sort w.r.t time
        iC = C.argsort()
        sC = C[iC]
        sS = S[iC]

        return sC, sS

    @property
    def wavelet_transform_delta(self):
        """Calculate the delta wavelet transform.

        Returns an array of the transform computed over the scales.
        """
        wavelet_func = self.wavelet.frequency  # wavelet as f(w_k, s)

        WK, S = jnp.meshgrid(self.w_k, self.scales)

        # compute Y_ over all s, w_k and sum over k
        norm = (2 * jnp.pi * S / self.dt) ** .5
        W_d = (1 / self.n) * jnp.sum(norm * wavelet_func(WK, S.T), axis=1)
        # N.B This W_d is 1D (defined only at n=0)
        return W_d

    @property
    def C_d(self):
        """Compute the parameter C_delta, used in
        reconstruction. See section 3.i of TC98.

        FIXME: this doesn't work. TC98 gives 0.776 for the Morlet
        wavelet with dj=0.125.
        """
        dj = self.dj
        dt = self.dt
        s = self.scales
        W_d = self.wavelet_transform_delta

        # value of the wavelet function at t=0
        Y_00 = self.wavelet.time(0).real

        real_sum = jnp.sum(W_d.real / s ** .5)
        C_d = real_sum * (dj * dt ** .5 / Y_00)
        return C_d


    def plot_power(self, ax=None, coi=True):
        """"Create a basic wavelet power plot with time on the
        x-axis, scale on the y-axis, and a cone of influence
        overlaid.
        """
        if not ax:
            fig, ax = plt.subplots()
        times = self.times
        scales = self.scales
        Time, Scale = jnp.meshgrid(times, scales)
        ax.contourf(Time, Scale, self.power, 100)
        ax.set_yscale('log')
        ax.grid(True)
        if coi:
            coi_time, coi_scale = self.coi
            ax.fill_between(x=coi_time,
                            y1=coi_scale,
                            y2=self.scales.max(),
                            color='gray',
                            alpha=0.3)

        ax.set_xlim(times.min(), times.max())
        return ax

########################################################################################################
# Tools for Wavelet Analysis
########################################################################################################


def find_s0(wavelet, dt):
    """Find the smallest resolvable scale by finding where the
        equivalent Fourier period is equal to 2 * dt. For a Morlet
        wavelet, this is roughly 1.
    """
    def f(s):
        return wavelet.fourier_period(s) - 2 * dt
    return scipy.optimize.fsolve(f, 1)[0]

def find_optimal_scales(s0, dt, dj, n):
        # Largest scale
        J = int((1 / dj) * math.log2(n * dt / s0))
        sj = s0 * 2 ** (dj * jnp.arange(0, J + 1))
        return sj



DEFAULT_WAVELET = morlet(w0=6)

def analyze(data, wavelet=DEFAULT_WAVELET, scales=None, dt=1., dj=0.125, 
    mask_coi=False, frequency=False, axis=-1):
    """Performs wavelet analysis on a dataset"""
    n = data.shape[axis]
    if scales is None:
        s0 = find_s0(wavelet, dt)
        scales = find_optimal_scales(s0, dt, dj, n)
    if frequency:
        scalogram = cwt_tc_frequency_jit(data, wavelet.frequency, scales, dt, axis)
    else:
        scalogram = cwt_tc_time_jit(data, wavelet.time, scales, dt, axis)
    scales = jnp.asarray(scales)
    return WaveletAnalysis(data=data, wavelet=wavelet, 
        dt=dt, dj=dj, mask_coi=mask_coi,
        frequency=frequency, axis=axis, scales=scales, 
        scalogram=scalogram)
