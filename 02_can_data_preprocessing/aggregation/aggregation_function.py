#####################################################################
# Copyright (C) 2025 ETH Zürich (ethz.ch)
# Chair of Information Management (im.ethz.ch; github.com/im-ethz)
# Bosch Lab at University of St. Gallen and ETH Zürich (iot-lab.ch)
#
# Authors: Robin Deuber, Kevin Koch, Patrick Langer, Martin Maritsch
#
# Licensed under the MIT License (the "License");
# you may only use this file in compliance with the License.
# You may obtain a copy of the License at
#
#         https://mit-license.org/
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#####################################################################

import numpy as np
from scipy.stats import skew, kurtosis, iqr

NUMERICAL_FUNCTIONS = {
    'mean': np.mean,
    'median': np.nanmedian,
    'std': np.std,
    'q5': lambda x: np.nanquantile(x, 0.05),
    'q95': lambda x: np.nanquantile(x, 0.95),
    'iqr': lambda x: iqr(x, nan_policy="omit"),
    'power': lambda x: np.nansum(x ** 2) / np.count_nonzero(x),
    'skewness': lambda x: skew(x, nan_policy="omit"),
    'kurtosis': lambda x: kurtosis(x, nan_policy="omit"),
    'n_sign_changes': lambda x: np.nansum(np.diff(np.sign(x)) != 0),
}


BINARY_FUNCTIONS = {
    'sum': np.nansum,
    'mean': np.nanmean,
    'std': np.std,
}

