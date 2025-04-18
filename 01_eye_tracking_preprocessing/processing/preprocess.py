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
import pandas as pd
from scipy.spatial.transform import Rotation as R

from processing.read_in_cam2world import read_in_cam2world
from processing.target_zones import get_target_zone_names


def get_valid_target_zone_names(data: pd.DataFrame):
    target_zone_names = get_target_zone_names()
    valid_target_zone_names = {}
    non_existing_target_zones = []

    for target_zone in target_zone_names:
        if target_zone in data.columns:
            valid_target_zone_names.update(
                {target_zone: target_zone_names[target_zone]["name"]}
            )
        else:
            non_existing_target_zones.append(target_zone_names[target_zone]["name"])

    return valid_target_zone_names, non_existing_target_zones


def preprocess(data: pd.DataFrame, directory_folder: str, confidence=0.1) -> pd.DataFrame:
    # Sort out all rows where all columns are empty.
    data.dropna(
        subset=data.columns.drop(["timestamp", "frame_number"]), inplace=True, how="all"
    )

    data = data[data["gaze_direction_confidence"] >= confidence]

    # One-hot encode the target zones and rename them to the names of the target zones.
    data = data.join(pd.get_dummies(data["target_zone"]))
    target_zone_names, non_existing_target_zones = get_valid_target_zone_names(data)

    data = data.rename(columns=target_zone_names, errors="raise")
    for non_existing_target_zone in non_existing_target_zones:
        data[non_existing_target_zone] = 0

    # Replace false state 144 in right and left eye state.
    data["right_eye_state"] = data["right_eye_state"].replace(144, -1)
    data["left_eye_state"] = data["left_eye_state"].replace(144, -1)

    # Units are wrong, transform from m to mm.
    data["right_eye_opening_mm"] = data["right_eye_opening_mm"] * 1000
    data["left_eye_opening_mm"] = data["left_eye_opening_mm"] * 1000

    # Transform coordinates from camera coordinate system into world coordinate system.
    cam2world = read_in_cam2world(directory_folder)
    cam2world_rot = cam2world[:3, :3]

    # Transform mideye origin to world coordinate system.
    mideye_world = (
        cam2world
        @ np.array(
            [
                data["mideye_origin_x"],
                data["mideye_origin_y"],
                data["mideye_origin_z"],
                np.ones(len(data)),
            ]
        )
    ).transpose()
    data.loc[:, "mideye_origin_x"] = mideye_world[:, 0]
    data.loc[:, "mideye_origin_y"] = mideye_world[:, 1]
    data.loc[:, "mideye_origin_z"] = mideye_world[:, 2]

    # Transform gaze vector to world coordinate system.
    gaze_direction_world = (
        cam2world_rot
        @ np.array(
            [
                data["gaze_direction_x"],
                data["gaze_direction_y"],
                data["gaze_direction_z"],
            ]
        )
    ).transpose()
    data.loc[:, "gaze_direction_x"] = gaze_direction_world[:, 0]
    data.loc[:, "gaze_direction_y"] = gaze_direction_world[:, 1]
    data.loc[:, "gaze_direction_z"] = gaze_direction_world[:, 2]

    # Transform head quaternions to world coordinate system.
    r_ccs = R.from_quat(
        np.array(
            [
                data["face_quat_x"],
                data["face_quat_y"],
                data["face_quat_z"],
                data["face_quat_w"],
            ]
        ).transpose()
    )
    head_wcs_rot = cam2world_rot @ r_ccs.as_matrix()
    r_wcs = R.from_matrix(head_wcs_rot)

    # Transform head rotation to euler angles.
    euler = r_wcs.as_euler("zxy").transpose()
    data["roll"] = euler[0]
    data["pitch"] = euler[1]
    data["yaw"] = euler[2]

    return data
