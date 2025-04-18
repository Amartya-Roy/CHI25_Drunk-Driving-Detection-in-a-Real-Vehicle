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

# Delta angle is defined such that delta will be positive if direction of change is in the direction of phi.
def get_delta_angle_arctan2(angle_start, angle_end):
    delta = angle_end - angle_start
    if delta > np.pi:
        delta -= 2 * np.pi
    if delta < -np.pi:
        delta += 2 * np.pi
    return delta

def calculate_velocity(data):
    window_width = 1
    step = int(np.ceil(float(window_width) / 2))

    speed = []
    speed_azimuth = []
    speed_elevation = []
    angle_change = []
    direction = []
    speed_roll = []
    speed_pitch = []
    speed_yaw = []
    speed_head = []
    speed_mideye_origin = []
    speed_mideye_origin_x = []
    speed_mideye_origin_y = []
    speed_mideye_origin_z = []

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

        # Invalid intervals; first speed of each sequence is set to 0.
        if start_pos == end_pos:
            speed.append(0)
            speed_azimuth.append(0)
            speed_elevation.append(0)
            angle_change.append(0)
            direction.append(0)
            speed_roll.append(0)
            speed_pitch.append(0)
            speed_yaw.append(0)
            speed_head.append(0)
            speed_mideye_origin.append(0)
            speed_mideye_origin_x.append(0)
            speed_mideye_origin_y.append(0)
            speed_mideye_origin_z.append(0)
            continue

        time = (data.index[end_pos] - data.index[start_pos]).total_seconds()

        # Calculate derivatives.
        delta_azimuth_temp = get_delta_angle_arctan2(data['azimuth'].iloc[start_pos], data['azimuth'].iloc[end_pos])
        delta_elevation_temp = get_delta_angle_arctan2(data['elevation'].iloc[start_pos], data['elevation'].iloc[end_pos])
        azimuth_deriv = delta_azimuth_temp / time
        elevation_deriv = delta_elevation_temp / time

        delta_roll = get_delta_angle_arctan2(data['roll'].iloc[start_pos], data['roll'].iloc[end_pos])
        delta_pitch = get_delta_angle_arctan2(data['pitch'].iloc[start_pos], data['pitch'].iloc[end_pos])
        delta_yaw = get_delta_angle_arctan2(data['yaw'].iloc[start_pos], data['yaw'].iloc[end_pos])
        speed_roll.append(delta_roll / time)
        speed_pitch.append(delta_pitch / time)
        speed_yaw.append(delta_yaw / time)
        speed_head.append(np.linalg.norm(np.asanyarray([delta_roll, delta_pitch, delta_yaw])) / time)

        # Calculate speed of mid eye origin.
        mid_eye_ampl = np.sqrt(
            ((data['mideye_origin_x'].iloc[end_pos] - data['mideye_origin_x'].iloc[start_pos]) ** 2) +
            ((data['mideye_origin_y'].iloc[end_pos] - data['mideye_origin_y'].iloc[start_pos]) ** 2) +
            ((data['mideye_origin_z'].iloc[end_pos] - data['mideye_origin_z'].iloc[start_pos]) ** 2)
        )
        speed_mideye_origin.append(mid_eye_ampl / time)
        speed_mideye_origin_x.append((data['mideye_origin_x'].iloc[end_pos] - data['mideye_origin_x'].iloc[start_pos]) / time)
        speed_mideye_origin_y.append((data['mideye_origin_y'].iloc[end_pos] - data['mideye_origin_y'].iloc[start_pos]) / time)
        speed_mideye_origin_z.append((data['mideye_origin_z'].iloc[end_pos] - data['mideye_origin_z'].iloc[start_pos]) / time)

        # Calculate speed.
        speed.append(np.sqrt((np.cos(data['elevation'].iloc[i-1]) ** 2) * (azimuth_deriv ** 2) + (elevation_deriv ** 2)))
        speed_azimuth.append(azimuth_deriv)
        speed_elevation.append(elevation_deriv)

        # Calculate the angular change as the norm of the change in azimuth and elevation direction.
        angle_change.append(np.sqrt((delta_azimuth_temp ** 2) + (delta_elevation_temp ** 2)))

        # Calculate direction as the angle between the horizontal line from the starting point and the position of the
        # end point.
        nominator = np.tan(data['elevation'].iloc[end_pos]) - np.tan(data['elevation'].iloc[start_pos])
        denominator = np.tan(data['azimuth'].iloc[end_pos]) - np.tan(data['azimuth'].iloc[start_pos])
        direction.append(np.arctan2(nominator, denominator))

    data['velocity'] = np.array(speed)
    data['velocity_azimuth'] = np.array(speed_azimuth)
    data['velocity_elevation'] = np.array(speed_elevation)

    data['angle_change'] = np.array(angle_change)

    data['direction'] = np.array(direction)

    data['velocity_roll'] = np.array(speed_roll)
    data['velocity_pitch'] = np.array(speed_pitch)
    data['velocity_yaw'] = np.array(speed_yaw)
    data['velocity_head'] = np.array(speed_head)

    data['velocity_mideye_origin'] = np.array(speed_mideye_origin)
    data['velocity_mideye_origin_x'] = np.array(speed_mideye_origin_x)
    data['velocity_mideye_origin_y'] = np.array(speed_mideye_origin_y)
    data['velocity_mideye_origin_z'] = np.array(speed_mideye_origin_z)

    return data
