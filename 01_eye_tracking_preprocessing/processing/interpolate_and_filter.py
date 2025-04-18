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

import pandas as pd
import numpy as np

def interpolate_and_filter(
    raw_data: pd.DataFrame) -> pd.DataFrame:

    non_float_cols = raw_data.select_dtypes(include="int").columns

    float_cols = raw_data.select_dtypes(include="float").columns

    raw_data = raw_data[raw_data["gaze_direction_confidence"] >= 0.01]

    frequency = 50.0
    target_index = pd.date_range(
        start=raw_data.index[0].floor("s"),
        end=raw_data.index[-1].ceil("s"),
        freq="%dus" % (1000000 / frequency),
    )
    raw_data = raw_data.reindex(
        index=raw_data.index.union(target_index).drop_duplicates()
    )

    raw_data.loc[:, float_cols] = raw_data.loc[:, float_cols].interpolate(
        method="time", limit=5, limit_direction="both"
    )
    raw_data.loc[:, non_float_cols] = raw_data.loc[:, non_float_cols].interpolate(
        method="nearest", limit=5, limit_direction="both"
    )
    raw_data = raw_data.reindex(target_index)

    # Ensure we have unit vectors again (numerical inaccuracies possible after filtering).
    gaze_direction_vector = ["gaze_direction_x", "gaze_direction_y", "gaze_direction_z"]
    raw_data[gaze_direction_vector] = raw_data[gaze_direction_vector].div(
        np.linalg.norm(raw_data[gaze_direction_vector], axis=1), axis=0
    )
    quat_vector = ["face_quat_x", "face_quat_y", "face_quat_z", "face_quat_w"]
    raw_data[quat_vector] = raw_data[quat_vector].div(
        np.linalg.norm(raw_data[quat_vector], axis=1), axis=0
    )

    return raw_data
