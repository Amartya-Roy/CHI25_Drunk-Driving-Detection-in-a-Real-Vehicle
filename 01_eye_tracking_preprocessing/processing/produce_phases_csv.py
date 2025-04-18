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

import os
import pandas as pd

# Produce_phases_csv reads the driving_exact csv file and filters the valid columns.
def produce_phases_csv(path: str) -> pd.DataFrame:
    path_suffix ="study_day/handwritten-notes/driving_exact.csv"
    driving_data = pd.read_csv(
        os.path.join(path, path_suffix), sep=","
    )

    driving_data["start_time"] = pd.to_datetime(
        driving_data["date"] + " " + driving_data["start_time"],
        format="%d.%m.%Y %H:%M:%S.%f",
    ).dt.tz_localize("Europe/Berlin")
    driving_data["end_time"] = pd.to_datetime(
        driving_data["date"] + " " + driving_data["end_time"],
        format="%d.%m.%Y %H:%M:%S.%f",
    ).dt.tz_localize("Europe/Berlin")

    driving_data.drop(columns=["notes", "date"], inplace=True)
    driving_data.rename(
        columns={
            "start_time": "start",
            "scenario_number": "variant",
            "end_time": "end",
        },
        inplace=True,
    )
    driving_data["intervention"] = False
    driving_data = driving_data.drop(driving_data.index[-1], inplace=False)
    driving_data = driving_data[driving_data["validity"] == 1]
    driving_data.drop(columns=["validity"], inplace=True)

    return driving_data
