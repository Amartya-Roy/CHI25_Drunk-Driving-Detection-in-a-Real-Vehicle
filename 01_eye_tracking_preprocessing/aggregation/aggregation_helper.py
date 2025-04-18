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

# Categorize eye movement and scenario column.
eye_categorization = {
    "FIXA": 0,
    "PURS": 1,
    "SACC": 2,
    "ISAC": 3,
    "MISSING": 4,
    "HPSO": 5,
    "IHPS": 6,
    "ILPS": 7,
    "LPSO": 8,
}
scenario_categorization = {"highway": 0, "rural": 1, "city": 2}

# Interpolate data to exactly desired frequency.
def interpolate_data(data: pd.DataFrame, binary_features: list[str]) -> pd.DataFrame:
    with pd.option_context("future.no_silent_downcasting", True):
        # Drop unused columns.
        df = data.copy()
        df = df.loc[df.index.dropna()]

        df["event+eye_movement_type+eventspec"] = df["event+eye_movement_type+eventspec"].replace(eye_categorization)
        df["groundtruth+scenario+"] = df["groundtruth+scenario+"].replace(scenario_categorization)
        df = df.astype({"groundtruth+scenario+": "int64", "event+eye_movement_type+eventspec": "int64"})

        # Panda automatically converts bool typed columns to object when reindexing to accommodate NaN values
        # this gives a warning in the interpolate method. Workaround: convert bool columns to int64 and then back
        # to bool after the interpolation.
        boolean_columns = df.dtypes[df.dtypes == "bool"].index
        df = df.astype({col: "int64" for col in boolean_columns})

        # Add target frequency index values.
        frequency = 50.0
        target_index = pd.date_range(
            start=df.index[0].ceil("s"),
            end=df.index[-1].floor("s"),
            freq="%dus" % (1000000 / frequency),
        )  # microsecs.

        df = df.reindex(index=df.index.union(target_index).drop_duplicates())

        # Interpolate all binary features and the categorized features with the nearest neighbour algorithm.
        integer_features = ["event+eye_movement_type+eventspec", "groundtruth+scenario+",
                            "groundtruth+phase+", "aoi+target_zone+"]
        eventspec_features = ["event+eye_movement_peak_vel+eventspec",
                              "event+eye_movement_avg_vel+eventspec",
                              "event+eye_movement_med_vel+eventspec",
                              "event+eye_movement_amp_given+eventspec",
                              "event+eye_movement_duration+eventspec"]
        df[
            binary_features + integer_features + eventspec_features
        ] = df[
            binary_features + integer_features + eventspec_features
        ].interpolate(
            method="nearest", limit=int(frequency)
        )

        columns_to_linearly_interpoloate = [value for value in df.columns
                                            if value not in (binary_features + integer_features + eventspec_features)]
        # Interpolate the remaining features linearly in time.
        df[columns_to_linearly_interpoloate] = df[columns_to_linearly_interpoloate].interpolate(method="time",
                                                                                                limit=int(frequency))

        # Keep data with exactly the targeted frequency.
        df = df.reindex(target_index)

        # Drop NaN values which can happen at the beginning and the ending.
        df.dropna(inplace=True, how='all')

        # Convert the binary features back to bool and the int features back to int64.
        df = df.astype({col: "bool" for col in binary_features} | {col: "int64" for col in integer_features})

        return df

