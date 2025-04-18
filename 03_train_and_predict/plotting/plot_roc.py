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

import numpy as np
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Patch

env_position_text = {"highway": 0.55, "rural": 0.615, "city": 0.6}
replace_scenario = {"highway": "Highway", "rural": "Rural", "city": "Urban"}

def roc_plot(ax, y_true, y_pred, subjects, label, color, prominent=True,
             xlabel=True, ylabel=True, sd=True, chance=True):
    if chance:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
                label='Chance', alpha=1)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100) 
    for t_id in np.unique(subjects):
        try:
            fpr, tpr, thresh = metrics.roc_curve(y_true[subjects == t_id], 
                                                 y_pred[subjects == t_id])
            auc = metrics.roc_auc_score(y_true[subjects == t_id], 
                                        y_pred[subjects == t_id])
        except ValueError as e:
            print(t_id, "exception was thrown: ", e)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # Mean line.
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=color,
            label=f'{label} (AUROC = {mean_auc:.2f}$\pm${np.std(aucs):.2f})',
            alpha=1 if prominent else .75, lw=2)

    # Mean std area.
    std_tpr = np.std(tprs, axis=0, ddof=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if sd:
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color,
                        alpha=.4 if prominent else .2, label=r'$\pm$1 SD')
            
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])

    text_color = '#000000'
    if ylabel:
        ax.set_ylabel('Sensitivity', color=text_color)
    if xlabel:
        ax.set_xlabel('1 - Specificity', color=text_color)
    
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_aspect('equal', 'box')
    return '%.2f$\pm$%.2f' % (mean_auc, std_auc)

def plot_roc(results, model_colors, scenario_colors):
    models = list(results.keys())
    fig, axs = plt.subplots(2, len(models), figsize=(9, 8), facecolor='white')
    plt.subplots_adjust(hspace=0.3)

    for column, model in enumerate(models):
        ax = axs[0, column]

        score = roc_plot(ax, results[model]["y_true"], results[model]["y_pred_proba"],
                         results[model]["subjects"], model, model_colors[model],
                         prominent=True, xlabel=True, ylabel=True, sd=True)

        text_box = AnchoredText('AUROC      \n', prop=dict(fontsize='10'),
                                frameon=True, loc='lower right', pad=0.38)
        plt.setp(text_box.patch, facecolor='white', edgecolor='#555555')
        ax.add_artist(text_box)

        ax.text(0.72, 0.04, score, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=10, color=model_colors[model], zorder=10)
        ax.set_title(model)
    
    labels = []
    lines = []
    
    for index, model in enumerate(models):
        if index == 2:
            labels.append('ROC $\pm$1 SD        ')
        else:
            labels.append('')
        lines.append(Patch(facecolor=model_colors[model], edgecolor='black'))
    
    labels.append('Chance')
    lines.append(matplotlib.lines.Line2D([0], [0], linestyle='--', lw=2,
                                         color='grey', alpha=1))

    for idx, env in enumerate(['highway', 'rural', 'city']):
        print(f"Env {env}")
        for column, model in enumerate(models):
            row = 1

            ax = axs[row, column]
            filter_scenarios = results[model]["scenarios"] == env
            score = roc_plot(ax, results[model]["y_true"][filter_scenarios],
                             results[model]["y_pred_proba"][filter_scenarios],
                             results[model]["subjects"][filter_scenarios],
                             "", scenario_colors[model][env],
                             prominent=False, xlabel=True, ylabel=True,
                             sd=False, chance=idx==0)

            if idx == 0:
                text_box = AnchoredText('AUROC                   \n\n\n',
                                        prop=dict(fontsize='10'), frameon=True,
                                        loc='lower right', pad=0.6)
                plt.setp(text_box.patch, facecolor='white', edgecolor='#555555')
                ax.add_artist(text_box)

            ax.text(env_position_text[env], 0.16-idx*0.06,
                    f"{replace_scenario[env]}: {score}", transform=ax.transAxes,
                    ha="left", fontsize=10, color=scenario_colors[model][env],
                    zorder=10)

    fig.subplots_adjust(left=0, top=1, bottom=0.12)

    lines = []
    labels = []

    m = model
    for key in scenario_colors[m].keys():      
        for index, model in enumerate(models):
            if index == 2:
                labels.append(f'ROC {replace_scenario[key]}         ')
            else:
                labels.append('')
            lines.append(Patch(facecolor=scenario_colors[model][key],
                               edgecolor='black'))

    labels.append('Chance')
    lines.append(matplotlib.lines.Line2D([0], [0], linestyle='--', lw=2,
                                         color='grey', alpha=1))

    for ax, letter in zip(axs[:, 0], list('ab')):
        ax.text(-0.2, 1.1, letter, transform=ax.transAxes, fontsize='18', weight='bold')

    plt.show()