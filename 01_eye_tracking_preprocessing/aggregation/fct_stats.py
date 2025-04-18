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

import datetime
import numpy as np
import pandas as pd
import math

from itertools import groupby
from scipy.stats import skew, kurtosis, iqr, entropy
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)

def get_stats(data: pd.DataFrame, key_suffix: str = None, epoch_width: int = 60) -> pd.DataFrame:

    results = {
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "q5": np.nan,
        "q95": np.nan,
        "iqr": np.nan,
        "power": np.nan,
        "skewness": np.nan,
        "kurtosis": np.nan,
        "n_sign_changes": np.nan,
    }

    if (len(data) > 0) and (not data.isna().all()):
        results["mean"] = np.mean(data)
        results["median"] = np.nanmedian(data)
        results["std"] = np.std(data)
        results["q5"] = np.nanquantile(data, 0.05)
        results["q95"] = np.nanquantile(data, 0.95)
        results["iqr"] = iqr(
            data, nan_policy="omit"
        )

        if np.count_nonzero(data) == 0:
            results["power"] = 0
        else:
            results["power"] = np.nansum([x**2 for x in data]) / np.count_nonzero(data)

        try:
            results["skewness"] = float(
                skew(data, nan_policy="omit")
            )
        except RuntimeWarning as e:
            print(f"Warning caught: {e}" + " cause by: " + key_suffix)

        try:
            results["kurtosis"] = kurtosis(
                data, nan_policy="omit"
            )
        except RuntimeWarning as e:
            print(f"Warning caught: {e}" + " caused by: " + key_suffix)

        results["n_sign_changes"] = np.nansum(
            np.diff(np.sign(data,)) != 0
        )

    if key_suffix is not None:
        results = {key_suffix + "+" + k: v for k, v in results.items()}

    results["agg+num_samples++"] = len(data)

    return results


def get_binary_event_stats(
    data: pd.DataFrame, target_zone_names: list[str], key_suffix: str = None, epoch_width: int = 60
) -> pd.DataFrame:

    results = {
        'duration': np.nan,
        'percentage_events': np.nan,
        'amplitude': np.nan,
        'event_count': np.nan,
    }

    if len(data) > 0:
        data = data.copy()
        number_all_type_events = (data['event+eye_movement_type+eventspec'].diff().fillna(0) != 0.0).sum()

        # Calculate durations per event and the target zone of the event.
        duration_events_list = []
        amplitudes_event_list = []
        ix = (data[key_suffix] * 1.0).diff().fillna(0)
        if (data[key_suffix] * 1.0).iloc[0] == 1.0:
            ix.iloc[0] = 1
        if (data[key_suffix] * 1.0).iloc[-1] == 1.0:
            ix.iloc[-1] = -1
        if -1.0 in ix.unique():
            event_times = list(zip(data.index[ix == 1], data.index[ix == -1]))
            for event_time in event_times:
                duration_events_list.append(
                    (
                        event_time[1] - event_time[0] - datetime.timedelta(seconds=0.02)
                    ).total_seconds()
                )
                amplitudes_event_list.append(
                    np.sum(np.abs(data.loc[event_time[0]:event_time[1]]['gaze+angle_change+velocity'].to_numpy())))

        # Calculate average duration, frequency and percentage of the eye movement type.
        if not duration_events_list:
            duration_events_list.append(0)
        if not amplitudes_event_list:
            amplitudes_event_list.append(0)

        total_duration_events = np.sum(duration_events_list)
        total_amplitude_events = np.sum(amplitudes_event_list)
        number_key_suffix_events = len([k for k, _ in groupby((data[key_suffix] * 1.0)) if k == 1])

        results['event_count'] = number_key_suffix_events

        if number_key_suffix_events > 0:
            results["duration"] = total_duration_events / number_key_suffix_events
            results['amplitude'] = total_amplitude_events / number_key_suffix_events
        else:
            results["duration"] = 0
            results['amplitude'] = 0

        if number_all_type_events > 0:
            results['percentage_events'] = number_key_suffix_events / number_all_type_events
        else:
            results['percentage_events'] = 0

    if key_suffix is not None:
            results = {key_suffix + "+" + k: v for k, v in results.items()}

    return results


def get_target_zone_stats(data, target_zone_names, key_suffix: str = None, epoch_width: int = 60):
    results = {}

    if len(data) > 0:
        data = data.copy()

        # Calculate the statistics of the entire window.
        number_all_regional_events_fixations = 0
        for region_number in target_zone_names:
            number_all_regional_events_fixations = number_all_regional_events_fixations + \
                                                   len([k for k, _ in groupby((data["aoi+target_zone+"] == region_number) &
                                                                              (data['event+FIXA+onehot'] == 1.0)) if k == 1])

        for region_number in target_zone_names:
            # Calculate durations of target zone events, considering only the
            # ones belonging to fixations.
            duration_events_list_fixations = []
            ix_fixations = (((data["aoi+target_zone+"] == region_number) &
                             (data['event+FIXA+onehot'] == 1.0)) * 1).diff().fillna(0)
            if ((data["aoi+target_zone+"] == region_number) & (data['event+FIXA+onehot'] == 1.0)).iloc[0] == 1.0:
                ix_fixations.iloc[0] = 1
            if ((data["aoi+target_zone+"] == region_number) & (data['event+FIXA+onehot'] == 1.0)).iloc[-1] == 1.0:
                ix_fixations.iloc[-1] = -1
            if -1.0 in ix_fixations.unique():
                event_times = list(zip(data.index[ix_fixations == 1], data.index[ix_fixations == -1]))
                for event_time in event_times:
                    duration_events_list_fixations.append(
                        (event_time[1] - event_time[0] - datetime.timedelta(seconds=0.02)).total_seconds())

            # Calculate average duration and percentage of the eye movement type.
            if not duration_events_list_fixations:
                duration_events_list_fixations.append(0)
            total_duration_events_fixations = np.sum(duration_events_list_fixations)
            number_region_events_fixations = len([k for k, _ in groupby((data["aoi+target_zone+"] == region_number) &
                                                                        (data['event+FIXA+onehot'] == 1.0)) if k == 1])

            duration_name = 'aoi+duration_fixations+' + str(target_zone_names[region_number] + "+")
            if number_region_events_fixations > 0:
                results[duration_name] = total_duration_events_fixations / number_region_events_fixations
            else:
                results[duration_name] = 0

            # Calculate the regional gaze event percentage, by using only the fixations.
            gaze_event_percentage = 'aoi+gaze_event_percentage+' + str(target_zone_names[region_number] + "+")
            if number_all_regional_events_fixations > 0:
                results[gaze_event_percentage] = number_region_events_fixations / number_all_regional_events_fixations
            else:
                results[gaze_event_percentage] = 0
    return results

def get_eventspec_stats(data, target_zone_names, key_suffix: str = None, epoch_width: int = 60):
    results = {}

    eye_categorization = {
        0: "FIXA",
        1: "PURS",
        2: "SACC",
        3: "ISAC",
        4: "MISSING",
        5: "HPSO",
        6: "IHPS",
        7: "ILPS",
        8: "LPSO",
    }

    eye_moment_type = data["event+eye_movement_type+eventspec"]
    eye_moment_type = eye_moment_type.replace(eye_categorization)

    metric_name = "_".join(key_suffix.split("+")[1].split("_")[-2:])

    if len(data) > 0:
        for movement_type in ["FIXA", "SACC"]:
            relevant_values = data[key_suffix][eye_moment_type == movement_type]
            filtered_values = relevant_values[relevant_values.diff().ne(0)]
            movement_type_dict = get_stats(filtered_values,
                                           ("event+" + movement_type + "+" + metric_name))

            for key, value in movement_type_dict.items():
                if isinstance(value, float) and math.isnan(value):
                    movement_type_dict[key] = 0

            keys_to_remove = [key for key in movement_type_dict if "agg+num_samples" in key]
            for key in keys_to_remove:
                del movement_type_dict[key]
                
            results.update(movement_type_dict)

    return results


