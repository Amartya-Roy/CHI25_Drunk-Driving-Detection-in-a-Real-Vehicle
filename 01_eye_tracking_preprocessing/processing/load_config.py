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

# The ProcessingConfig class is used to store the parameters for the processing pipeline.
class ProcessingConfig:
    def __init__(
        self,
        raw_input_directory: str,
        preprocessed_output_directory: str,
        probands_selected: list[str],
        selected_phases: list[int],
        selected_scenarios: list[str],
        remodnav_args: list[str],
        confidence: float = 0.01,
        run_probands_in_parallel: bool = False
    ) -> None:
        self.raw_input_directory = raw_input_directory
        self.preprocessed_output_directory = preprocessed_output_directory
        self.probands_selected = probands_selected
        self.run_probands_in_parallel = run_probands_in_parallel
        self.selected_phases = selected_phases
        self.selected_scenarios = selected_scenarios
        self.confidence = confidence
        self.remodnav_args = remodnav_args


# Load config parameters from yaml file.
def load_config(filename: str) -> ProcessingConfig:
    with open(filename, "r") as yamlfile:
        cfg_processing = yaml.load(yamlfile, Loader=yaml.FullLoader)

    keys = ["raw_input_directory", "preprocessed_output_directory",
            "probands_selected", "run_probands_in_parallel", "selected_phases",
            "selected_scenarios", "confidence", "remodnav_args"]

    # Get the values of the given keys.
    cfg_processing_dict = {key: cfg_processing.get(key) for key in keys}

    # Keys for which the retrieved value is None.
    none_keys = []
    for key, value in cfg_processing_dict.items():
        if value is None:
            none_keys.append(key)

    # Print the keys with None values if there are any.
    if none_keys:
        print("The following values could not be found in the config file:\n", none_keys)

    # Read in processing configs.
    return ProcessingConfig(**cfg_processing)
