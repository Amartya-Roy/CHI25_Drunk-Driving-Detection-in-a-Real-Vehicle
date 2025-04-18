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
import os

date_format = "%d.%m.%Y %H:%M:%S.%f"

def add_bac_level(data: pd.DataFrame, directory_folder: str) -> pd.DataFrame:
    """
    Adds the blood alcohol concentration (BAC) to the data.
    The BAC is measured during phase and scenario changes on the
    study day and is interpolated to the data timestamps.
    """

    handwritten_notes_folder = os.path.join(
        directory_folder, "study_day/handwritten-notes/"
    )

    study_day_date_df = pd.read_csv(
        os.path.join(handwritten_notes_folder, "general.csv"), sep=","
    )
    study_day_date = study_day_date_df[
        study_day_date_df["var_name"] == "study_day_date"
    ]["value"].item()

    driving_data = pd.read_csv(
        os.path.join(handwritten_notes_folder, "driving_exact.csv"), sep=","
    )

    driving_data["start_time"] = pd.to_datetime(
        study_day_date + " " + driving_data["start_time"], format=date_format
    )
    driving_data["end_time"] = pd.to_datetime(
        study_day_date + " " + driving_data["end_time"], format=date_format
    )

    BAC_data = pd.read_csv(
        os.path.join(handwritten_notes_folder, "BAC_driving.csv"), sep=","
    )

    # We want to get the beginning timestamp for each phase and scenario,
    # as there might be multiple rows in the driving data we take the min value.
    # For the after measurement we take the max end value at the end of each phase.
    measurement_times = (pd.concat([driving_data.groupby(["phase", "scenario"])
                                   .start_time.min()
                                   .reset_index()
                                   .start_time,
                                    driving_data.groupby(["phase"]).end_time.max().reset_index().end_time])
                         .sort_values().reset_index(drop=True))
    BAC_data["timestamp"] = measurement_times

    BAC_data = BAC_data.set_index("timestamp")
    BAC_data.index = pd.to_datetime(BAC_data.index)
    BAC_data.index = BAC_data.index.tz_localize("CET")
    BAC_data = BAC_data.reindex(index=BAC_data.index.union(data.index))
    BAC_data.index = pd.to_datetime(BAC_data.index)
    BAC_data = BAC_data.sort_index()
    BAC_data = BAC_data.drop(columns=["measurement", "phase"])

    BAC_data["BAC"] = pd.to_numeric(BAC_data["BAC"], errors="coerce")

    BAC_data["BAC"] = BAC_data["BAC"].interpolate(method="time", limit_direction="both")

    BAC_data = BAC_data.infer_objects(copy=False)
    BAC_data = BAC_data.reindex(data.index)

    data["BAC"] = BAC_data

    return data
