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

# Check if the phases and scenarios that were selected actually exist for this proband.
def check_phases_scenarios(data_phases, selected_phases, selected_scenarios):
    selected_phases_checked = []
    selected_scenarios_checked = []
    for phase in selected_phases:
        data_phase = data_phases[data_phases['phase'] == phase]
        if not data_phase.empty:
            selected_phases_checked.append(phase)
            for scenario in selected_scenarios:
                data_scenario = data_phase[data_phase['scenario'] == scenario]
                if not data_scenario.empty and (scenario not in selected_scenarios_checked):
                    selected_scenarios_checked.append(scenario)

    return selected_phases_checked, selected_scenarios_checked
