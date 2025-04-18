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

from joblib import Parallel, delayed
from processing import load_config
from processing import process_subject
import os

class ProcessingPipeline:
    def __init__(self, config_file: str) -> None:
        # Load basic configs from .yaml file.
        self.config = load_config(config_file)

    def run(self):
        Parallel(n_jobs=min(self.config.n_jobs, len(self.config.alcohol_subjects)), verbose=10)(
            delayed(process_subject)(
                subject,
                os.path.join(self.config.data_directory, 'drive_' + str(subject) + '/'),
                os.path.join(self.config.data_output_directory, 'drive_' + str(subject) + '/'),
                freq=self.config.freq,
                is_ref=False, reusing=self.config.reusing_old_df) for subject in
            self.config.alcohol_subjects)

        if self.config.set_reference_phase:
            Parallel(n_jobs=min(self.config.n_jobs, len(self.config.reference_placebo_subjects)), verbose=10)(
                delayed(process_subject)(
                    subject,
                    os.path.join(self.config.data_directory, 'drive_' + str(subject) + '/'),
                    os.path.join(self.config.data_output_directory, 'drive_' + str(subject) + '/'),
                    freq=self.config.freq,
                    is_ref=True, reusing=self.config.reusing_old_df,
                    ref_phase_to=self.config.reference_phase_set_to)
                for subject in self.config.reference_placebo_subjects)
        else:
            Parallel(n_jobs=min(self.config.n_jobs, len(self.config.reference_placebo_subjects)), verbose=10)(
                delayed(process_subject)(
                    subject,
                    os.path.join(self.config.data_directory, 'drive_' + str(subject) + '/'),
                    os.path.join(self.config.data_output_directory, 'drive_' + str(subject) + '/'),
                    freq=self.config.freq,
                    is_ref=False, reusing=self.config.reusing_old_df) for subject in self.config.reference_placebo_subjects)
