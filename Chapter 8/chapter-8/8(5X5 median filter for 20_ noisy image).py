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

def apply_median_filter(image, filter_size):
    filtered_image = cv2.medianBlur(image, filter_size)

    return filtered_image

# Read the image from file
image_path = "C:\\DIP\\Dataset\\5.3.01.tiff"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add 20% salt and pepper noise to the image
noise_percentage_20 = 0.20
noisy_image_20 = add_salt_and_pepper_noise(image.copy(), noise_percentage_20)

# Apply a 5x5 Median Filter to denoise the image
filter_size_5 = 5
denoised_image_20 = apply_median_filter(noisy_image_20, filter_size_5)

# Display the original image, the noisy image, and the denoised image
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image (20%)', noisy_image_20)
cv2.imshow('Denoised Image (20%)', denoised_image_20)
cv2.waitKey(0)
cv2.destroyAllWindows()
