import numpy as np
from scipy.signal import medfilt2d

# Input 6x6 grayscale image values from the user
print("Enter the values for the 6x6 grayscale image :\n\n")
image_6x6 = np.zeros((6, 6), dtype=np.uint8)
for i in range(6):
    row_input = input(f" Enter values of row {i+1}: ")
    row_values = row_input.split()
    for j in range(6):
        value = int(row_values[j])
        image_6x6[i, j] = value

# Apply median filter to obtain 4x4 grayscale image
image_4x4 = medfilt2d(image_6x6, kernel_size=3)[1::2, 1::2]

# Print the resulting 4x4 grayscale image
print("4x4 Grayscale Image after applying median filter:\n\n")
print(image_4x4)
