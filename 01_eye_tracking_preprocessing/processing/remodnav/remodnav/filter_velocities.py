import numpy as np


def filter_velocities(velocities, max_vel, print_warning):
    filtered_velocities = [float(0)]
    for vel in velocities:
        if vel > max_vel:  # deg/s
            # ignore very fast velocities
            """if print_warning:
                print('Computed velocity exceeds threshold. Inappropriate filter setup? ' +
                      str(vel) + ' > ' + str(max_vel) + ' deg/s')"""
            vel = filtered_velocities[-1]
        filtered_velocities.append(vel)
    velocities = np.array(filtered_velocities)

    return velocities
