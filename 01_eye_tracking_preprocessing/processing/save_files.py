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

import os
import pickle
import pandas as pd

# save_files writes the results of processing into csv and pkl files.
def save_files(data: pd.DataFrame, output_directory: str, folder: str, data_phases: pd.DataFrame,
               selected_phases: list[int], selected_scenarios: list[str], selected_phase_times: list[pd.Timestamp],
               selected_scenario_times: list[pd.Timestamp]):

    # Define directory for save.
    directory_save = os.path.join(
        output_directory, folder, "ircam")

    os.makedirs(directory_save, exist_ok=True)

    try:
        data.to_pickle(directory_save + '/' + folder + '.pkl')
    except:
        print(folder, data.info())
    data_phases.to_csv(directory_save + '/phases_' + folder + '.csv')
    data_phases.to_pickle(directory_save + '/phases_' +
                          folder + '.pkl')
    with open(directory_save + '/selected_phases.pkl', 'wb') as f:
        pickle.dump(selected_phases, f, protocol=2)
    with open(directory_save + '/selected_scenarios.pkl', 'wb') as f:
        pickle.dump(selected_scenarios, f, protocol=2)
    with open(directory_save + '/selected_phase_times.pkl', 'wb') as f:
        pickle.dump(selected_phase_times, f, protocol=2)
    with open(directory_save + '/selected_scenario_times.pkl', 'wb') as f:
        pickle.dump(selected_scenario_times, f, protocol=2)

    print('Successfully processed and saved the data from proband ' + folder)
