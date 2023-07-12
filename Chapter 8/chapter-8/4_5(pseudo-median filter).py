import numpy as np
import matplotlib.pyplot as plt

def pseudo_median_filter(image):
    height, width = image.shape
    filtered_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            window = image[max(0, i - 1):min(height, i + 2),
                           max(0, j - 1):min(width, j + 2)]
            median_approximation = (np.max(window) + np.min(window)) // 2
            filtered_image[i, j] = median_approximation

    return filtered_image

# Input 6x6 grayscale image values from the user
print("Enter the values for the 6x6 grayscale image (separated by spaces):")
image_6x6 = np.zeros((6, 6), dtype=np.uint8)
for i in range(6):
    row_input = input(f"Enter values for row {i+1}: ")
    row_values = row_input.split()
    for j in range(6):
        value = int(row_values[j])
        image_6x6[i, j] = value

# Apply pseudo median filtering
filtered_image = pseudo_median_filter(image_6x6)




# Extract the central 4x4 region
image_4x4 = filtered_image[1:5, 1:5]
print("Resulted 4x4 image value after applying filtering\n\n")
print(image_4x4)

# Display the original 6x6 image, the filtered 6x6 image, and the resulting 4x4 image
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image_6x6, cmap='gray')
axs[0].set_title('Original 6x6 Image')
axs[0].axis('off')
axs[1].imshow(filtered_image, cmap='gray')
axs[1].set_title('Filtered 6x6 Image')
axs[1].axis('off')
axs[2].imshow(image_4x4, cmap='gray')
axs[2].set_title('Resulting 4x4 Image')
axs[2].axis('off')

plt.show()
