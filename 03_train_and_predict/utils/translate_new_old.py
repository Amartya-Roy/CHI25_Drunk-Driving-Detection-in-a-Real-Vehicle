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

import numpy as np

def translate_new_old(model_infos, config):
    groups = model_infos["data"]["groundtruth+id++"].unique()

    all_results = dict()
    for group in np.unique(groups):
        if group in config["selected_participants"]["reference"]:
            continue
        if group in config["selected_participants"]["placebo"]:
            continue
        
        results = dict()
        results["y_true"] = model_infos["data"]["y_test"][model_infos["data"]["groundtruth+id++"] == group]
        results["y_pred_proba"] = model_infos["data"]["y_proba_test"][model_infos["data"]["groundtruth+id++"] == group]
        results["y_pred"] = model_infos["data"]["y_pred_test"][model_infos["data"]["groundtruth+id++"] == group]
        results["y_orig"] = model_infos["data"]["y_orig_test"][model_infos["data"]["groundtruth+id++"] == group]
        results["scenario"] = model_infos["data"]["scenarios"][model_infos["data"]["groundtruth+id++"] == group]
        results["coefs"] = model_infos[group]["coefs"]

        all_results[group] = results

    all_results["features"] = model_infos['features']
    all_results["name"] = model_infos['name']
    all_results["window_size"] = model_infos['window_size'] 

    return all_results
    