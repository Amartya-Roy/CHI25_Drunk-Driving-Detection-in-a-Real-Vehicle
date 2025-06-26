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
import pandas as pd
import copy
from timeit import default_timer as timer
from joblib import Parallel, delayed

from aggregation.aggregation_helper import interpolate_data
from aggregation.crop_data_aggregation import crop_data_aggregation
from aggregation.add_phase_scenario_columns import add_phase_scenario_columns
from aggregation.load_config import load_config
from aggregation.load_data import load_data

class AggregationPipeline:
    def __init__(self, config_file: str) -> None:
        # Load basic configs from .yaml file.
        self.config = load_config(config_file)

    def save_one_participant_csv_safely(self, proband_id: str):
        try:
            data_agg = self.save_one_participant_csv(proband_id)
            return data_agg
        except Exception as e:
            print(proband_id + ":")
            print(f"Exception occurred: {e}")

    def save_one_participant_csv(self, proband_id: str):
        config = copy.deepcopy(self.config)
        print('----------------------------------------')
        print('Proband: ', proband_id)

        directory_save = os.path.join(config.data_directory_processed, proband_id, 'ircam/')

        filename = os.path.join(directory_save, f"{proband_id}_merged.pkl")
        file_exists_already = os.path.exists(filename)

        if (not config.enforce_recalculation) and file_exists_already:
            print(f"The files for participant {proband_id} will be reloaded.")
            data_agg = pd.read_pickle(filename)
            return data_agg

        # Load data from processed files.
        directory_processed = config.data_directory_processed

        data, data_phases = load_data(
            directory_processed, proband_id, 'aggregation', config)

        # Ensure all eye movement are present in the data.
        eye_movement_binary = ['event+FIXA+onehot', 'event+SACC+onehot']
        for x in eye_movement_binary:
            if x not in data.columns:
                data[x] = 0

        # Interpolate data to constant frequency.
        data = interpolate_data(data, config.binary_features)

        # Crop data for faster aggregation.
        data = crop_data_aggregation(
            data, data_phases, config.selected_phases, config.selected_scenarios, 0)

        # Add phase, scenario and variant columns.
        data = add_phase_scenario_columns(
            data, data_phases, config.selected_phases)

        data = data.copy()
        data['groundtruth+id++'] = proband_id
        data.rename(columns={'groundtruth+BAC+': 'groundtruth+BAC++'}, inplace=True)

        # Save data to parquet and pickle files.
        os.makedirs(directory_save, exist_ok=True)
        data.to_parquet(os.path.join(directory_save, f"{proband_id}_merged.parquet"))
        data.to_pickle(filename)

        return data

    def save_all_participants_csv(self, data_agg_all, directory_data):
        print('----------------------------------------')
        print('----------------------------------------')
        print('Summarizing all probands')
        print('Number of probands: ' + str(len(data_agg_all)))

        # Concat list of all probands to one dataframe.
        data_agg_all = pd.concat(data_agg_all)
        data_agg_all.sort_index(inplace=True)

        # Create the destination directory if it does not exist.
        directory_path = os.path.join(directory_data, 'ircam')
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        directory_save = os.path.join(directory_path, "all_probands.parquet")
        data_agg_all.to_parquet(directory_save)
        print('Saved in: ', directory_save)
        return

    def run(self) -> pd.DataFrame:
        # Define directories.
        directory_processed = self.config.data_directory_processed

        proband_ids = [proband_id for proband_id in os.listdir(
            directory_processed) if not proband_id.startswith('.') and proband_id[-3:] in self.config.probands_selected]

        start_time = timer()

        if self.config.multi_cores is False or len(proband_ids) == 1:
            print("Not using multiple cores.")
            data_agg_all = []
            for proband_id in proband_ids:
                data_agg_all.append(self.save_one_participant_csv(str(proband_id)))
            self.save_all_participants_csv(
                data_agg_all, self.config.data_directory_processed)
        elif self.config.multi_cores is True:
            print("Using multiple cores.")
            with Parallel(n_jobs=min(19, len(proband_ids)), verbose=101, backend='multiprocessing') as parallel:
                data_agg_all = parallel(delayed(self.save_one_participant_csv_safely)(
                    str(proband_id)) for proband_id in proband_ids)

            self.save_all_participants_csv(
                data_agg_all, self.config.data_directory_processed)
        else:
            print('Error with multi_cores setting in script!')
            exit()

        end_time = timer()
        process_time = end_time - start_time
        print(
            f'Finished aggregation in {process_time} seconds.')

        return data_agg_all
