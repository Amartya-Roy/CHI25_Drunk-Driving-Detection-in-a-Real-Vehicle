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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

pipe_lasso = Pipeline([
    ('scale', StandardScaler()),
    ('clf', LogisticRegression(penalty='l1', solver='saga', class_weight='balanced'))
])

pipe_ridge = Pipeline([
    ('scale', StandardScaler()),
    ('clf', LogisticRegression(penalty='l2', solver='saga', class_weight='balanced'))
])

pipe_elasticnet = Pipeline([
    ('scale', StandardScaler()),
    ('clf', LogisticRegression(penalty='elasticnet', l1_ratio=0.5,
                               solver='saga', class_weight='balanced'))
])

pipe_SVC = Pipeline([
    ('scale', StandardScaler()),
    ('clf', SVC(probability=True))
])

pipe_GB = Pipeline([
    ('scale', StandardScaler()),
    ('clf', GradientBoostingClassifier())
])

pipe_mlp = Pipeline([
    ('scale', StandardScaler()),
    ('clf', MLPClassifier())
])

pipe_RandomForest = Pipeline([
    ('scale', StandardScaler()),
    ('clf', RandomForestClassifier())
])

