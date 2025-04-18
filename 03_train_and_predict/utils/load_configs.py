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


def load_configs(config_file: str) -> dict[str, any]:
    with open(config_file, 'r') as yamlfile:
        cfg_prediction = yaml.load(yamlfile, Loader=yaml.FullLoader)

    config = {}
    config["data_directory"] = cfg_prediction["data_directory"]
    config["use_parallel_processing"] = cfg_prediction['use_parallel_processing']
    config["verbose"] = cfg_prediction['verbose']
    config["dmc_features"] = cfg_prediction['dmc_features']
    config["can_features"] = cfg_prediction['can_features']
    config["window_length"] = cfg_prediction['window_length']
    config["num_cores"] = cfg_prediction['num_cores']
    config["use_dmc"] = cfg_prediction['use_dmc']
    config["use_can"] = cfg_prediction['use_can']
    config["use_placebo"] = cfg_prediction['use_placebo']
    config["use_reference"] = cfg_prediction['use_reference']
    config["models"] = cfg_prediction['models']

    treatment_participants = cfg_prediction['treatment_participants']
    placebo_participants = cfg_prediction['placebo_participants']
    reference_participants = cfg_prediction['reference_participants']
    selected_participants = {"treatment": treatment_participants,
                             "reference": reference_participants,
                             "placebo": placebo_participants}
    config["selected_participants"] = selected_participants

    return config
