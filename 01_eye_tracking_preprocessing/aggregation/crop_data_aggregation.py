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

# Crop data to selected phases and scenarios.
def crop_data_aggregation(
    raw_data: pd.DataFrame,
    data_phases: pd.DataFrame,
    selected_phases: list[int],
    selected_scenarios: list[str],
    epoch_width: int,
) -> pd.DataFrame:
    data_phases = data_phases.copy()

    raw_selected_data = []
    for index, row in data_phases.iterrows():
        if row["phase"] in selected_phases:
            if row["scenario"] in selected_scenarios:
                scenario_start = row["start"]
                scenario_end = row["end"] - datetime.timedelta(seconds=epoch_width)
                raw_selected_data.append(raw_data[scenario_start:scenario_end])

    return pd.concat(raw_selected_data)
