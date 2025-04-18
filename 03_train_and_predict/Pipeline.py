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

from utils.load_configs import load_configs
from utils.load_data import load_data
from utils.model_training_evaluation import train_LOSO_safely
from utils.pipelines import pipe_lasso
from utils.translate_new_old import translate_new_old
from utils.evaluate import evaluate

from plotting.main_plotting import main_plotting

class Pipeline:
    def __init__(self, config_path=''):
        self.config = load_configs(config_path if config_path != '' else 'config_prediction.yml')

    def load_data(self):
        self.data, self.core_features = load_data(config=self.config)

    def train(self):
        self.result_dfs = {}
        self.model_infos = {}

        for model in self.config["models"]:
            if model == "Early Warning":
                y_column = "y_EW"
            if model == "Above Limit":
                y_column = "y_AL"

            self.model_infos[model] = train_LOSO_safely(
                self.data, pipe_lasso, y_column, self.core_features, model, self.config)

            self.result_dfs[model] = translate_new_old(
                self.model_infos[model], self.config)

    def evaluate(self):
        self.evaluation_overall = {}
        self.evaluation_scenario = {}

        for key in self.result_dfs:
            if self.config["verbose"]:
                print("Model", key)
            self.evaluation_overall[key] = evaluate(
                self.model_infos[key], self.config)

        for key in self.result_dfs:
            if self.config["verbose"]:
                print("Model per scenario", key)
            self.evaluation_scenario[key] = evaluate(
                self.model_infos[key], self.config, col_analysis_factor='scenarios')

    def plot_results(self):
        if self.config["verbose"]:
            main_plotting(self.result_dfs, self.config)
