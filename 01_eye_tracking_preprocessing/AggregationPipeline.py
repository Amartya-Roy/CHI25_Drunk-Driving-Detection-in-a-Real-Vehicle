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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#####################################################################

"""Pipeline to merge preprocessed eye tracking files."""

import os
import pandas as pd
from aggregation.load_config import load_config
from aggregation.load_data import load_data


class AggregationPipeline:
    """Simplified aggregation for eye tracking data."""

    def __init__(self, config_file: str) -> None:
        self.config = load_config(config_file)

    def run(self) -> pd.DataFrame:
        """Load all preprocessed files and combine them into a single parquet."""
        data_all = []
        for proband_id in self.config.probands_selected:
            df = load_data(self.config.data_directory_processed, str(proband_id))
            df["groundtruth+id++"] = str(proband_id)
            rename_cols = {
                "groundtruth+phase+": "groundtruth+phase++",
                "groundtruth+scenario+": "groundtruth+scenario++",
                "groundtruth+variant+": "groundtruth+variant++",
                "groundtruth+BAC+": "groundtruth+BAC++",
            }
            df.rename(columns={k: v for k, v in rename_cols.items() if k in df.columns}, inplace=True)
            data_all.append(df)

        if not data_all:
            raise RuntimeError("No data found for selected participants")

        data_agg = pd.concat(data_all).sort_index()
        output_dir = os.path.join(self.config.data_directory_processed, "ircam")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "all_probands.parquet")
        data_agg.to_parquet(output_file)
        print("Saved aggregated data to", output_file)
        return data_agg
