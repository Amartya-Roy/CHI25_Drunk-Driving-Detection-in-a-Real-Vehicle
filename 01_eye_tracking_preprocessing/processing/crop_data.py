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

from typing import Tuple
import pandas as pd
import warnings

# Crop data to selected phases and scenarios.
def crop_data(
    raw_data: pd.DataFrame,
    data_phases: pd.DataFrame,
    selected_phases: list[int],
    selected_scenarios: list[str],
) -> Tuple[pd.DataFrame, list[pd.Timestamp], list[pd.Timestamp]]:

    # Filter the phases based on the selected phases and scenarios.
    filtered_phases = data_phases[
        data_phases["phase"].isin(selected_phases)
        & data_phases["scenario"].isin(selected_scenarios)
    ]

    # Ensure that start and end columns are in datetime format.
    filtered_phases.loc[:, "start"] = filtered_phases["start"].apply(lambda x: pd.to_datetime(x))
    filtered_phases.loc[:, "end"] = filtered_phases["end"].apply(lambda x: pd.to_datetime(x))

    # Check that scenario start is before scenario end.
    start_times = filtered_phases["start"].apply(lambda x: x.to_pydatetime().time())
    end_times = filtered_phases["end"].apply(lambda x: x.to_pydatetime().time())
    if (start_times > end_times).any():
        warnings.warn("Some scenarios have start times after end times")

    raw_selected_data = pd.concat(
        [
            raw_data.between_time(
                phase.start.to_pydatetime().time(), phase.end.to_pydatetime().time()
            )
            for index, phase in filtered_phases.iterrows()
        ]
    )

    # Get the phase and scenario times separately to visualize them in the plots.
    selected_phase_times = []
    selected_scenario_times = []
    for phase in selected_phases:
        phase_scenarios = data_phases[data_phases["phase"] == phase]
        selected_phase_times.append(phase_scenarios["start"].iloc[0])
        selected_phase_times.append(phase_scenarios["end"].iloc[-1])
        for index, row in phase_scenarios.iterrows():
            selected_scenario_times.append(row["start"])
            selected_scenario_times.append(row["end"])

    return raw_selected_data, selected_phase_times, selected_scenario_times
