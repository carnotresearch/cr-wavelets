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

import matplotlib.pyplot as plt
import numpy as np

def plot_decomposition(fig, axes, coeffs, stem=False):
    """
    """
    levels = len(coeffs) - 1
    for i, (ax, coeff) in enumerate(zip(axes, coeffs)):
        if i == 0:
            title = f'cA{levels}'
        else:
            title = f'cD{levels - i+1}'
        if stem:
            ax.stem(coeff, markerfmt='.', linefmt='C0--')
        else:
            ax.plot(coeff)
        ax.set_title(title)


def plot_2_decompositions(fig, axes, coeffs1, coeffs2,
    label1='Original', label2='Reconstructed', stem=False):
    levels = len(coeffs1) - 1
    for i, (ax, c1, c2) in enumerate(zip(axes, coeffs1, coeffs2)):
        if i == 0:
            title = f'cA{levels}'
        else:
            title = f'cD{levels - i+1}'
        if stem:
            n = len(c1)
            x = np.arange(n)
            ax.stem(x, c1, label=label1, markerfmt='.', linefmt='C0--')
            ax.stem(x, c2, label=label2, markerfmt='.', linefmt='C1--')
        else:
            ax.plot(c1, label=label1)
            ax.plot(c2, label=label2)
        ax.set_title(title)
        ax.legend(loc=1)
