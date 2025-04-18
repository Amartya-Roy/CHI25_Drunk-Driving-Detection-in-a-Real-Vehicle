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
import os
from xml.dom import minidom

def read_in_cam2world(directory_folder):
    """
    Gets the cam2world coordinates from the camera calibration file.
    """
    base_directory = directory_folder + '/study_day/ircam/'
    directories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    # Filter directories that start with "Calibration_".
    calibration_folder = [folder for folder in directories if folder.startswith('Calibration_')][0]
    calibration_file_path = base_directory + calibration_folder + '/intermediate/'
    
    calibration_document = minidom.parse(calibration_file_path + 'CalibrationData.xml')
    items = calibration_document.getElementsByTagName('Camera_WRT_World')
    cam2world_values = np.array(items[0].firstChild.nodeValue.replace('[', '').replace(']', '').split(', ')).astype(
        float).reshape((4, 3))
    cam2world = np.zeros((4, 4))
    cam2world[3, 3] = 1
    cam2world[:cam2world_values.shape[0], :cam2world_values.shape[1]] = cam2world_values
    cam2world = cam2world.transpose()

    return cam2world
