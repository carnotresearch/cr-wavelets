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


import numpy as np

def plot_2_signals(fig, ax, sig1, sig2,
    fs, label1='Original', label2='Reconstructed',
    xlabel='Time (sec)', ylabel='Amplitude'):
    sig1 = np.asarray(sig1)
    sig2 = np.asarray(sig2)
    n = len(sig1)
    t = np.arange(n) * (1/fs)
    ax.plot(t, sig1, label=label1)
    ax.plot(t, sig2, label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=1)
