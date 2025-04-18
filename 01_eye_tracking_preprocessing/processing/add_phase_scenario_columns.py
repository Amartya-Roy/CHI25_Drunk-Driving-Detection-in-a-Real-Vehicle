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

import warnings

def add_phase_scenario_columns(data, data_phases, selected_phases):

    for _, phase in data_phases.iterrows():
        if phase["phase"] not in selected_phases:
            continue
        indexer = data[
            (data.index >= phase["start"]) & (data.index <= phase["end"])
        ].index
        if len(indexer) == 0:
            warnings.warn("Driving section between " + str(phase["start"]) + " and " + str(phase["end"]) + " is empty.")
            continue
        data.loc[indexer, ["phase", "scenario", "variant"]] = phase[
            ["phase", "scenario", "variant"]
        ].values

    data["phase"] = data["phase"].astype(str)

    data["phase"] = data["phase"].astype(float).astype(int)
    data["variant"] = data["variant"].astype(float).astype(int)


    return data
