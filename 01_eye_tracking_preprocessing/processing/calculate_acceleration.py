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

def calculate_acceleration(data):
    window_width = 1
    step = int(np.ceil(float(window_width) / 2))

    acceleration_list = []
    acceleration_r_list = []
    acceleration_azimuth_list = []
    acceleration_elevation_list = []

    acceleration_roll_list = []
    acceleration_pitch_list = []
    acceleration_yaw_list = []
    acceleration_head_list = []

    acceleration_mideye_origin = []
    acceleration_mideye_origin_x = []
    acceleration_mideye_origin_y = []
    acceleration_mideye_origin_z = []

    for i in range(len(data)):
        # Get initial interval.
        if step == window_width:
            start_pos = i - step
            end_pos = i
        else:
            start_pos = i - step
            end_pos = i + step

        if start_pos < 0:
            start_pos = i

        if end_pos > len(data) - 1:
            end_pos = i

        # Invalid intervals; first acceleration of each sequence is set to 0.
        if start_pos == end_pos:
            acceleration_list.append(0)
            acceleration_r_list.append(0)
            acceleration_azimuth_list.append(0)
            acceleration_elevation_list.append(0)
            acceleration_roll_list.append(0)
            acceleration_pitch_list.append(0)
            acceleration_yaw_list.append(0)
            acceleration_head_list.append(0)
            acceleration_mideye_origin.append(0)
            acceleration_mideye_origin_x.append(0)
            acceleration_mideye_origin_y.append(0)
            acceleration_mideye_origin_z.append(0)
            continue

        # Calculate derivatives.
        time = (data.index[end_pos] - data.index[start_pos]).total_seconds()
        acceleration_azimuth = (data['velocity_azimuth'].iloc[end_pos] - data['velocity_azimuth'].iloc[start_pos]) / time
        acceleration_elevation = (data['velocity_elevation'].iloc[end_pos] - data['velocity_elevation'].iloc[start_pos]) / time

        # Acceleration of head rotation.
        acceleration_roll_list.append((data['velocity_roll'].iloc[end_pos] - data['velocity_roll'].iloc[start_pos]) / time)
        acceleration_pitch_list.append((data['velocity_pitch'].iloc[end_pos] - data['velocity_pitch'].iloc[start_pos]) / time)
        acceleration_yaw_list.append((data['velocity_yaw'].iloc[end_pos] - data['velocity_yaw'].iloc[start_pos]) / time)
        acceleration_head_list.append((data['velocity_head'].iloc[end_pos] - data['velocity_head'].iloc[start_pos]) / time)

        # Acceleration of mid eye origin.
        acceleration_mideye_origin.append(
            (data['velocity_mideye_origin'].iloc[end_pos] - data['velocity_mideye_origin'].iloc[start_pos]) / time)
        acceleration_mideye_origin_x.append(
            (data['velocity_mideye_origin_x'].iloc[end_pos] - data['velocity_mideye_origin_x'].iloc[start_pos]) / time)
        acceleration_mideye_origin_y.append(
            (data['velocity_mideye_origin_y'].iloc[end_pos] - data['velocity_mideye_origin_y'].iloc[start_pos]) / time)
        acceleration_mideye_origin_z.append(
            (data['velocity_mideye_origin_z'].iloc[end_pos] - data['velocity_mideye_origin_z'].iloc[start_pos]) / time)

        # Calculate accelerations in different directions.
        acceleration_r_direction = -(data['velocity_elevation'].iloc[i] ** 2) - (data['velocity_azimuth'].iloc[i] ** 2) * (
                np.cos(data['elevation'].iloc[i]) ** 2)
        acceleration_azimuth_direction = acceleration_azimuth * np.cos(data['elevation'].iloc[i]) - 2 * data[
            'velocity_elevation'].iloc[i] * data['velocity_azimuth'].iloc[i] * np.sin(data['elevation'].iloc[i])
        acceleration_elevation_direction = acceleration_elevation + (data['velocity_azimuth'].iloc[i] ** 2) * np.sin(
            data['elevation'].iloc[i]) * np.cos(data['elevation'].iloc[i])

        # Append accelerations to lists.
        acceleration_list.append(np.linalg.norm(
            np.array([acceleration_r_direction, acceleration_azimuth_direction, acceleration_elevation_direction])))
        acceleration_r_list.append(acceleration_r_direction)
        acceleration_azimuth_list.append(acceleration_azimuth_direction)
        acceleration_elevation_list.append(acceleration_elevation_direction)

    data['acceleration'] = np.array(acceleration_list)
    data['acceleration_r'] = np.array(acceleration_r_list)
    data['acceleration_azimuth'] = np.array(acceleration_azimuth_list)
    data['acceleration_elevation'] = np.array(acceleration_elevation_list)

    data['acceleration_roll'] = np.array(acceleration_roll_list)
    data['acceleration_pitch'] = np.array(acceleration_pitch_list)
    data['acceleration_yaw'] = np.array(acceleration_yaw_list)
    data['acceleration_head'] = np.array(acceleration_head_list)

    data['acceleration_mideye_origin'] = np.array(acceleration_mideye_origin)
    data['acceleration_mideye_origin_x'] = np.array(acceleration_mideye_origin_x)
    data['acceleration_mideye_origin_y'] = np.array(acceleration_mideye_origin_y)
    data['acceleration_mideye_origin_z'] = np.array(acceleration_mideye_origin_z)

    return data
