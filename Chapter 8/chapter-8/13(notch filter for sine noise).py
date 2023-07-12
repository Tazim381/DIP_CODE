import numpy as np
import cv2

def add_sine_noise(image, frequency):
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    noise = 1 + np.sin(x / 3 + y / 5) * frequency
    noisy_image = np.clip(image * noise, 0, 255).astype(np.uint8)
    return noisy_image

def notch_filter(image, notches):
    rows, cols = image.shape
    filtered_image = np.fft.fft2(image)

    for notch in notches:
        center_x, center_y, radius = notch
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        filter_mask = np.logical_and(distance > radius - 5, distance < radius + 5)
        filter_mask = np.logical_not(filter_mask)
        filtered_image *= np.fft.fftshift(filter_mask)

    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.abs(filtered_image).astype(np.uint8)
    return filtered_image

image_path = "C:\\DIP\\Dataset\\5.3.01.tiff"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Add sine noise
noisy_image = add_sine_noise(image, frequency=0.1)

# Define notch locations
notch_centers = [(image.shape[1] // 2, image.shape[0] // 2)]
notch_radius = 30

# Apply notch filter to remove sine noise
notches = [(center[0], center[1], notch_radius) for center in notch_centers]
filtered_image = notch_filter(noisy_image, notches)

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Filtered Image", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
