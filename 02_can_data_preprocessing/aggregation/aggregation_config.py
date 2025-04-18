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

import yaml

class AggregationConfig:
    def __init__(
            self,
            data_directory: str,
            relative_subject_output_directory: str,
            data_output_directory: str,
            freq: int,
            n_jobs: int,
            subjects: list,
            aggregation_sizes: list,
            reusing_old_df: bool = False,
    ) -> None:
        self.data_directory = data_directory
        self.relative_subject_output_directory = relative_subject_output_directory
        self.data_output_directory = data_output_directory
        self.freq = freq
        self.n_jobs = n_jobs
        self.subjects = subjects
        self.reusing_old_df = reusing_old_df
        self.aggregation_sizes = aggregation_sizes


def load_config(filename: str) -> AggregationConfig:
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return AggregationConfig(
        data_directory=config['data_directory'],
        relative_subject_output_directory=config['relative_subject_output_directory'],
        data_output_directory=config['data_output_directory'],
        freq=config['frequency'],
        n_jobs=config['n_jobs'],
        subjects=config['subjects'],
        aggregation_sizes=config['aggregation_sizes'],
        reusing_old_df=config['reusing_old_df']
    )

