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

import datetime
import pandas as pd
import pytz

# This function loads the raw DMC data from the csv file and returns a pandas dataframe.
def load_file(filename: str) -> pd.DataFrame:

    # Skip the first row for because of incorrect format.
    rows_to_skip = [1]

    df = pd.read_csv(
        filename,
        sep=";",
        skiprows=rows_to_skip,
        dtype={
            "timestamp": int,
            "frame_number": int,
            "filename": str,
            "0_face_x": int,
            "0_face_y": int,
            "0_face_width": int,
            "0_face_height": int,
            "0_face_confidence": float,
            "0_face_quat_w": float,
            "0_face_quat_x": float,
            "0_face_quat_y": float,
            "0_face_quat_z": float,
            "0_face_trans_x": float,
            "0_face_trans_y": float,
            "0_face_trans_z": float,
            "0_face_yaw": float,
            "0_face_pitch": float,
            "0_face_roll": float,
            "0_mideye_origin_x": float,
            "0_mideye_origin_y": float,
            "0_mideye_origin_z": float,
            "0_mideye_origin_confidence": float,
            "0_gaze_direction_x": float,
            "0_gaze_direction_y": float,
            "0_gaze_direction_z": float,
            "0_gaze_direction_confidence": float,
            "0_gaze_direction_source": int,
            "0_target_zone": int,
            "0_left_eye_opening_mm": float,
            "0_left_eye_opening_percent": float,
            "0_left_eye_confidence": float,
            "0_left_eye_state": int,
            "0_right_eye_opening_mm": float,
            "0_right_eye_opening_percent": float,
            "0_right_eye_confidence": float,
            "0_right_eye_state": int,
            "0_drowsiness": int,
            "0_drowsinessTime_ms": int,
            "0_inattention": int,
            "0_inattentionTime_ms": int,
            "0_accumulatedInattention": int,
            "0_accumulatedInattentionTime_ms": int,
            "LeftEyeOutercorner_V1_x": int,
            "LeftEyeOutercorner_V1_y": int,
            "LeftEyeOutercorner_V1_attribute": int,
            "LeftEyeInnercorner_V1_x": int,
            "LeftEyeInnercorner_V1_y": int,
            "LeftEyeInnercorner_V1_attribute": int,
            "RightEyeOutercorner_V1_x": int,
            "RightEyeOutercorner_V1_y": int,
            "RightEyeOutercorner_V1_attribute": int,
            "RightEyeInnercorner_V1_x": int,
            "RightEyeInnercorner_V1_y": int,
            "RightEyeInnercorner_V1_attribute": int,
            "LeftMouthcorner_V1_x": int,
            "LeftMouthcorner_V1_y": int,
            "LeftMouthcorner_V1_attribute": int,
            "RightMouthcorner_V1_x": int,
            "RightMouthcorner_V1_y": int,
            "RightMouthcorner_V1_attribute": int,
            "LeftNostrilSill_V1_x": int,
            "LeftNostrilSill_V1_y": int,
            "LeftNostrilSill_V1_attribute": int,
            "RightNostrilSill_V1_x": int,
            "RightNostrilSill_V1_y": int,
            "RightNostrilSill_V1_attribute": int,
        },
    )

    # Read the first row just to get the correct timestamp value.
    first_row = pd.read_csv(filename, sep=";", nrows=1)
    timestamp = filename.split("/")[-1].split(".")[0]
    timestamp = datetime.datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
    df["timestamp"] = df["timestamp"] - first_row["timestamp"][0]

    df.index = pd.to_datetime(
        (df["timestamp"] + timestamp.timestamp() * 1000).map(int), unit="ms"
    )
    df.index = df.index.tz_localize(pytz.utc).tz_convert(pytz.timezone("CET"))
    df.index.names = ["time"]

    # Last column is empty and face_userid does not contain useful data.
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    try:
        df.drop(columns=['0_face_userid'], inplace=True)
    except KeyError:
        pass

    # Remove all "0_" before the column names.
    df.columns = df.columns.str.replace("0_", "", 1)

    df.drop(columns=["filename"], inplace=True)

    return df
