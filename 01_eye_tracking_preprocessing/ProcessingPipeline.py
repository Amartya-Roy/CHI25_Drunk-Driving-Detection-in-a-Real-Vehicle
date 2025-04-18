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
from joblib import Parallel, delayed
from processing.add_blood_biometrics import add_bac_level
from processing.add_eye_movement import add_eye_movement
from processing.add_phase_scenario_columns import add_phase_scenario_columns
from processing.calculate_acceleration import calculate_acceleration
from processing.calculate_spherical_coordinates import calculate_spherical_coordinates
from processing.calculate_velocity import calculate_velocity
from processing.check_phases_scenarios import check_phases_scenarios
from processing.crop_data import crop_data
from processing.interpolate_and_filter import interpolate_and_filter
from processing.load_config import load_config
from processing.load_raw_file import load_file
from processing.preprocess import preprocess
from processing.produce_phases_csv import produce_phases_csv
from processing.rad_to_deg import rad_to_deg
from processing.remodnav.remodnav.remodnav import remodnav
from processing.renaming_conventions import renaming_convention_dict
from processing.save_files import save_files

# ProcessingPipeline defines the high-level steps for loading,
# preprocessing and saving the data of the selected probands.
class ProcessingPipeline:
    def __init__(self, config_file: str) -> None:
        # Load basic configs from .yaml file.
        self.config = load_config(config_file)

    def run(self):
        folders = sorted(os.listdir(self.config.raw_input_directory))
        folders = [
            folder
            for folder in folders
            # Select folder if last 3 digits (proband_id) are in the selected config list.
            if folder[-3:] in self.config.probands_selected
        ]

        print(f"Processing {len(folders)} probands")

        if self.config.run_probands_in_parallel:
            with Parallel(
                n_jobs=min(32, len(folders)), verbose=51, backend="multiprocessing"
            ) as parallel:
                parallel(delayed(self.run_proband_safely)(folder) for folder in folders)
        else:
            for folder in folders:
                self.run_proband(folder)

    # Read in the data of one proband from all available .csv files.
    def load_data(self, directory_folder: str) -> pd.DataFrame:
        path_suffix = "study_day/ircam/"

        directory_ircam = os.path.join(
            directory_folder, path_suffix
        )
        raw_data = []
        for file in os.listdir(directory_ircam):
            if (
                not file.startswith(".")
                and file.endswith(".csv")
                and os.path.isfile(os.path.join(directory_ircam, file))
            ):
                file_data = load_file(os.path.join(directory_ircam, file))
                raw_data.append(file_data)

        # Concat the data from all .csv files to one data frame and sort according to date.
        raw_data = pd.concat(raw_data)
        raw_data.sort_index(inplace=True)
        return raw_data

    def preprocess_data(
        self, raw_data: pd.DataFrame, folder: str, directory_folder: str
    ):
        # Read in the times of the data phases and check if all requested data phases are available.
        data_phases = produce_phases_csv(directory_folder)

        raw_data = interpolate_and_filter(raw_data)

        # Add the blood alcohol concentration data.
        raw_data = add_bac_level(raw_data, directory_folder)

        selected_phases_checked, selected_scenarios_checked = check_phases_scenarios(data_phases, self.config.selected_phases,
                                                                                      self.config.selected_scenarios)

        # Crop data such that only the data of the requested scenarios and phases are left.
        raw_selected_data, selected_phase_times, selected_scenario_times = crop_data(
            raw_data, data_phases, selected_phases_checked, selected_scenarios_checked
        )

        # Preprocess the data and calculate gaze features.
        data = preprocess(
            raw_selected_data,
            directory_folder,
            confidence=self.config.confidence,
        )

        data = calculate_spherical_coordinates(data)

        # Filter data and determine eye movement types with the REMODNAV algorithm.
        remodnav_args = self.config.remodnav_args
        data, data_events = remodnav(data, remodnav_args)

        # Add eye movement types to data and one-hot encode them.
        add_eye_movement(data, data_events)
        data = data.join(pd.get_dummies(data["eye_movement_type"]))

        # Calculate velocity, acceleration.
        data = calculate_velocity(data)
        data = calculate_acceleration(data)

        # Transform all data from radians to degree.
        data = rad_to_deg(data)

        # Add the phase and the scenario of each data point to the data.
        data = add_phase_scenario_columns(data, data_phases, selected_phases_checked)

        data.rename(columns=renaming_convention_dict, inplace=True)

        # Save data to a csv file.
        save_files(
            data,
            self.config.preprocessed_output_directory,
            folder,
            data_phases,
            selected_phases_checked,
            selected_scenarios_checked,
            selected_phase_times,
            selected_scenario_times,
        )

    # Process a single proband.
    def run_proband(self, folder: str):
        directory_folder = os.path.join(self.config.raw_input_directory, folder)
        data = self.load_data(directory_folder)
        self.preprocess_data(data, folder, directory_folder)

    # Wrapper around run_proband to catch exceptions.
    def run_proband_safely(self, folder: str):
        try:
            print("Running proband: ", folder)
            self.run_proband(folder)
        except Exception as e:
            print(folder)
            print(f"Exception occurred: {e}")
