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

from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    average_precision_score, f1_score, recall_score, precision_score, \
    matthews_corrcoef, roc_auc_score
import numpy as np

scores = {"auroc": roc_auc_score, "prcauc": average_precision_score,
          "b_acc": balanced_accuracy_score, "acc": accuracy_score,
          "f1_weighted": f1_score, "recall": recall_score,
          "precision": precision_score, "mcc": matthews_corrcoef}

def calc_score(score, y_test, y_pred_test, y_proba_test):
    if score == "auroc" or score == "prcauc":
        score_result = scores[score](y_test, y_proba_test)
    elif score == "f1_weighted":
        score_result = scores[score](y_test, y_pred_test, average="weighted")
    else:
        score_result = scores[score](y_test, y_pred_test)
    return score_result

def evaluate(model_infos, config, col_fold = 'groundtruth+id++', col_analysis_factor = None):
    fold_ids = model_infos["data"][col_fold].unique()
    
    results = dict()
    if not col_analysis_factor:
        for score in scores.keys():
            score_result = []
            for fold in fold_ids:
                y_test = model_infos["data"][model_infos["data"][col_fold]==fold]['y_test']
                # Ensure that fold is not reference or placebo participant.
                if y_test.nunique() == 1:
                    continue
                y_pred_test = model_infos["data"][model_infos["data"][col_fold]==fold]['y_pred_test']
                y_proba_test = model_infos["data"][model_infos["data"][col_fold]==fold]['y_proba_test']
                score_result.append(calc_score(score, y_test, y_pred_test, y_proba_test))
            results[score] = score_result
    else:
        for score in scores.keys():
            for col_analysis_factor_goup in model_infos["data"][col_analysis_factor].unique():
                score_result = []
                for fold in fold_ids:
                    y_test = model_infos["data"][(model_infos["data"][col_fold]==fold) & (model_infos["data"][col_analysis_factor]==col_analysis_factor_goup)]['y_test']
                    # Ensure that fold is not reference or placebo participant.
                    if y_test.nunique() == 1:
                        continue
                    y_pred_test = model_infos["data"][(model_infos["data"][col_fold]==fold) & (model_infos["data"][col_analysis_factor]==col_analysis_factor_goup)]['y_pred_test']
                    y_proba_test = model_infos["data"][(model_infos["data"][col_fold]==fold) & (model_infos["data"][col_analysis_factor]==col_analysis_factor_goup)]['y_proba_test']
                    score_result.append(calc_score(score, y_test, y_pred_test, y_proba_test))
                results[col_analysis_factor_goup + "_" + score] = score_result
  
    if config["verbose"]:
        for key in results.keys():
            if key != "driver":
                print("{}: M {:.2f} +/- SD {:.2f}".format(key, np.mean(results[key]), np.std(results[key])))
    
    return results