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

import pandas as pd
from utils.pipelines import pipe_lasso, pipe_RandomForest
from sklearn.base import clone


def train_sklearn_LR_lasso(X_train, y_train, X_test):
    clf = clone(pipe_lasso)
    clf_fitted = clf.fit(X_train, y_train)
    y_pred_proba_train = clf_fitted.predict_proba(X_train)[:, 1]
    y_pred_proba_test = clf_fitted.predict_proba(X_test)[:, 1]

    coef = pd.DataFrame({"Feature": X_train.columns.tolist() + ["intercept"],
                         "Coefficients": [*clf_fitted["clf"].coef_[0],
                                          *clf_fitted["clf"].intercept_]})

    return y_pred_proba_train, y_pred_proba_test, coef

def train_sklearn_general(X_train, y_train, X_test):
    features = X_train.columns.tolist()

    clf = clone(pipe_RandomForest)
    clf_fitted = clf.fit(X_train, y_train)
    y_pred_proba_train = clf_fitted.predict_proba(X_train)[:, 1]
    y_pred_proba_test = clf_fitted.predict_proba(X_test)[:, 1]

    coef = pd.DataFrame({"Feature": features,
                         "Coefficients": [0] * X_train.shape[1]})

    return y_pred_proba_train, y_pred_proba_test, coef



