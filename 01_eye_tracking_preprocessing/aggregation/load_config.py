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
        data_directory_processed: str,
        probands_selected: list[int],
        multi_cores: bool,
        enforce_recalculation: bool,
        aggregation_sizes: list[int],
        step_size: int,
        selected_phases: list[int],
        selected_scenarios: list[str],
        framerate: int,
        numerical_features: list[str],
        binary_features: list[str],
        single_eye_movement_features: list[str],
    ) -> None:
        self.data_directory_processed = data_directory_processed
        self.probands_selected = probands_selected
        self.multi_cores = multi_cores
        self.enforce_recalculation = enforce_recalculation
        self.aggregation_sizes = aggregation_sizes
        self.step_size = step_size
        self.selected_phases = selected_phases
        self.selected_scenarios = selected_scenarios
        self.framerate = framerate
        self.numerical_features = numerical_features
        self.binary_features = binary_features
        self.single_eye_movement_features = single_eye_movement_features


# Load config parameters from yaml file.
def load_config(filename: str) -> AggregationConfig:
    with open(filename, "r") as yamlfile:
        cfg_aggregation = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return AggregationConfig(
        cfg_aggregation["data_directory_processed"],
        cfg_aggregation["probands_selected"],
        cfg_aggregation["multi_cores"],
        cfg_aggregation["enforce_recalculation"],
        cfg_aggregation["aggregation_sizes"],
        cfg_aggregation["step_size"],
        cfg_aggregation["selected_phases"],
        cfg_aggregation["selected_scenarios"],
        cfg_aggregation["framerate"],
        cfg_aggregation["numerical_features"],
        cfg_aggregation["binary_features"],
        cfg_aggregation["single_eye_movement_features"],
    )
