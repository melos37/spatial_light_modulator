
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:44:22 2022

@author: Hy-Q @NBI, KU
"""

import numpy as np
import matplotlib.pyplot as plt

# Camera-picture size
size_a = 768
size_b = 1024
zoom_factor = 4

# Effective camera-picture size after zooming
zoomed_size_a = size_a // zoom_factor
zoomed_size_b = size_b // zoom_factor

# Physical size of the image sensor in millimeters
sensor_width_mm = 10
sensor_height_mm = 10

# Calculate the camera pixel size
pixel_size_a = sensor_width_mm / zoomed_size_a
pixel_size_b = sensor_height_mm / zoomed_size_b

# Load the numpy array of images
image_array = np.load(r'D:\PUK\PUK results\Periods_=_10.0_to_80.0_Amplitude_=_60.5_to_60.5.npy')

# Container for spot distances
spot_distances = []

# Initialize previous spot position for spot 1
prev_spot1 = None

# Iterate over each image in the array starting from the second image
for image in image_array[1:]:
    # Find the pixel with maximum intensity for spot 1
    spot1_pixel = np.unravel_index(np.argmax(image), image.shape)

    if prev_spot1 is None:
        prev_spot1 = spot1_pixel

    # Exclude the region around spot 1 for spot 2 detection
    mask = np.zeros(image.shape, dtype=bool)
    mask[max(0, prev_spot1[0] - 10):prev_spot1[0] + 11,
         max(0, prev_spot1[1] - 10):prev_spot1[1] + 11] = True
    image[mask] = 0

    # Find the pixel with maximum intensity for spot 2
    spot2_pixel = np.unravel_index(np.argmax(image), image.shape)

    # Calculate the distance between the spots in nanometers
    distance = np.sqrt((prev_spot1[0] - spot2_pixel[0]) ** 2 + (prev_spot1[1] - spot2_pixel[1]) ** 2)
    distance_nm = distance * pixel_size_a * zoom_factor * 10  # Convert from mm to nm
    spot_distances.append(distance_nm)

    # Update previous spot position for the next iteration
    prev_spot1 = spot1_pixel

# Generate frequency values
frequency = np.linspace(10, 80, len(spot_distances))

# Perform least squares linear fit
fit_coeffs = np.polyfit(frequency, spot_distances, 1)
fit_line = np.polyval(fit_coeffs, frequency)

# Plot distance as a function of frequency with the fit line
plt.plot(frequency, spot_distances, color='blue', label='Data')
plt.plot(frequency, fit_line, color='red', linestyle='--', label='Fit Line')
plt.xlabel("Frequency")
plt.ylabel("Distance (nm)")
plt.title("Distance between Spot 1 and Spot 2")
plt.legend()
plt.show()
