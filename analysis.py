import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the numpy array of images
image_array = np.load('D:\PUK results\Periods_=_70.0_to_70.0_Amplitude_=_0.0_to_127.5.npy')

# Container for spot intensities
spot1_intensities = []
spot2_intensities = []

# Initialize previous spot position for spot 1
prev_spot1 = None

# Iterate over each image in the array
for image in image_array:
    # Find the pixel with maximum intensity for spot 1
    spot1_pixel = np.unravel_index(np.argmax(image), image.shape)

    if prev_spot1 is None:
        prev_spot1 = spot1_pixel

    # Measure intensity of spot 1
    spot1_intensity = image[prev_spot1]
    spot1_intensities.append(spot1_intensity)

    # Exclude the region around spot 1 for spot 2 detection
    mask = np.zeros(image.shape, dtype=bool)
    mask[max(0, prev_spot1[0] - 10):prev_spot1[0] + 11,
         max(0, prev_spot1[1] - 10):prev_spot1[1] + 11] = True
    image[mask] = 0

    # Find the pixel with maximum intensity for spot 2
    spot2_pixel = np.unravel_index(np.argmax(image), image.shape)

    # Measure intensity of spot 2
    spot2_intensity = image[spot2_pixel]
    spot2_intensities.append(spot2_intensity)

# Perform least squares polynomial fit
degree = 5  # Adjust the degree of the polynomial fit as desired
spot1_fit = np.polyfit(np.arange(len(spot1_intensities)), spot1_intensities, degree)
spot2_fit = np.polyfit(np.arange(len(spot2_intensities)), spot2_intensities, degree)

# Generate smoothed lines
spot1_smooth = np.polyval(spot1_fit, np.arange(len(spot1_intensities)))
spot2_smooth = np.polyval(spot2_fit, np.arange(len(spot2_intensities)))

# Plot only the smoothed lines
plt.plot(spot1_smooth, label='Spot 1 ')
plt.plot(spot2_smooth, label='Spot 2 ')
plt.xlabel("Amplitude")
plt.ylabel("Intensity")
plt.title("Spot Intensities")
plt.legend()
plt.show()

# Save spot intensities as CSV
data = {
    'Spot 1 Smooth': spot1_smooth,
    'Spot 2 Smooth': spot2_smooth
}
df = pd.DataFrame(data)
df.to_csv('spot_intensities.csv', index=False)
