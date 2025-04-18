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

import glob
import pandas as pd

def merge_with_scenario(df: pd.DataFrame, data_folder: str, is_ref=False, ref_phase_to=1) -> pd.DataFrame:
    scenario_file = glob.glob(data_folder + "/study_day/handwritten-notes/driving_exact.csv")
    if len(scenario_file) == 0:
        raise IOError("No scenario file found")

    scenarios = pd.read_csv(scenario_file[0], index_col=None, header=0,
                            names=['phase', 'scenario', 'scenario_number', 'date', 'start_time', 'end_time',
                                   'validity', 'notes'])
    scenarios.dropna(subset=['date', 'start_time', 'end_time'], inplace=True)

    df['notes'] = ''

    for idx, row in scenarios.iterrows():
        start_time = pd.to_datetime(row['date'] + ' ' + row['start_time'], format='%d.%m.%Y %H:%M:%S.%f').tz_localize(
            'Europe/Zurich')
        end_time = pd.to_datetime(row['date'] + ' ' + row['end_time'], format='%d.%m.%Y %H:%M:%S.%f').tz_localize(
            'Europe/Zurich')
        in_scenario = (df.index >= start_time) & (df.index <= end_time)
        df.loc[in_scenario, 'phase'] = row['phase']
        if is_ref:
            # For reference data, the phase is all 1 ("sober").
            df.loc[in_scenario, 'phase'] = ref_phase_to
        df.loc[in_scenario, 'scenario'] = row['scenario']
        df.loc[in_scenario, 'scenario_number'] = row['scenario_number']
        df.loc[in_scenario, 'validity'] = row['validity']
        df.loc[in_scenario, 'notes'] = row['notes']

    df.dropna(subset=['scenario'], inplace=True)
    return df


def fix_the_timestamp(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # Time synchronization of the logging device was somtimes corrupted.
    timestamps = df[column_name].copy()
    idx = len(df) - 2
    while idx >= 0:
        if timestamps[idx].year != 2023:
            gap = timestamps[idx + 1] - timestamps[idx]
            incorrect = idx
            while incorrect >= 0 and timestamps[incorrect].year != 2023:
                timestamps[incorrect] += gap
                incorrect -= 1
            idx = incorrect
        idx -= 1
    df[column_name] = timestamps
    return df
