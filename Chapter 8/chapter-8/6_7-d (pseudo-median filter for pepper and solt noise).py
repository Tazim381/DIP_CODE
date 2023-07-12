import cv2
import numpy as np

def add_salt_and_pepper_noise(image, noise_percentage):
    height, width = image.shape[:2]
    num_pixels = int(noise_percentage * height * width)

    # Add salt noise
    salt_coords = [np.random.randint(0, d, num_pixels) for d in image.shape[:2]]
    image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    pepper_coords = [np.random.randint(0, d, num_pixels) for d in image.shape[:2]]
    image[pepper_coords[0], pepper_coords[1]] = 0

    return image

def apply_pseudo_median_filter(image):
    height, width = image.shape[:2]
    filtered_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            window = image[max(0, i - 1):min(height, i + 2),
                           max(0, j - 1):min(width, j + 2)]
            sorted_values = np.sort(window, axis=None)
            median_value = sorted_values[len(sorted_values) // 2]
            filtered_image[i, j] = median_value

    return filtered_image

# Read the image from file
image_path =  "C:\\DIP\\Dataset\\5.3.01.tiff"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add 5% salt and pepper noise to the image
noise_percentage_5 = 0.05
noisy_image_5 = add_salt_and_pepper_noise(image.copy(), noise_percentage_5)

# Add 10% salt and pepper noise to the image
noise_percentage_10 = 0.10
noisy_image_10 = add_salt_and_pepper_noise(image.copy(), noise_percentage_10)

# Add 20% salt and pepper noise to the image
noise_percentage_20 = 0.20
noisy_image_20 = add_salt_and_pepper_noise(image.copy(), noise_percentage_20)

# Apply the Pseudo-Median filter to denoise the images
denoised_image_5 = apply_pseudo_median_filter(noisy_image_5)
denoised_image_10 = apply_pseudo_median_filter(noisy_image_10)
denoised_image_20 = apply_pseudo_median_filter(noisy_image_20)

# Display the original image, the noisy images, and the denoised images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image (5%)', noisy_image_5)
cv2.imshow('Denoised Image (5%)', denoised_image_5)
cv2.imshow('Noisy Image (10%)', noisy_image_10)
cv2.imshow('Denoised Image (10%)', denoised_image_10)
cv2.imshow('Noisy Image (20%)', noisy_image_20)
cv2.imshow('Denoised Image (20%)', denoised_image_20)
cv2.waitKey(0)
cv2.destroyAllWindows()
