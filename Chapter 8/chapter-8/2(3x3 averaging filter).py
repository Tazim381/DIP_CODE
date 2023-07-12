import numpy as np
from scipy.ndimage import convolve

# Input 6x6 grayscale image values from the user
print("Enter the values for the 6x6 grayscale image :")
image_6x6 = np.zeros((6, 6), dtype=np.uint8)
for i in range(6):
    row_input = input(f"Enter values for row {i+1}: ")
    row_values = row_input.split()
    for j in range(6):
        value = int(row_values[j])
        image_6x6[i, j] = value

# Define the 3x3 averaging filter kernel
kernel = np.ones((3, 3), dtype=np.float32) / 9.0

# Apply the averaging filter to obtain the 4x4 grayscale image
image_4x4 = convolve(image_6x6, kernel)[1::2, 1::2]

# Print the resulting 4x4 grayscale image
print("4x4 Grayscale Image after applying 3*3 Averaging filter :")
print(image_4x4)
