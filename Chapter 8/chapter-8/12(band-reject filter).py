import numpy as np
import cv2
from scipy import ndimage

def add_sine_noise(image, frequency):
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    noise = np.sin((x + y / 1.5) * frequency)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def band_reject_filter(image, frequency, bandwidth):
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    filter_mask = np.logical_or(np.abs(x - cols / 2) < bandwidth / 2, np.abs(y - rows / 2) < bandwidth / 2)
    filter_mask = np.logical_not(filter_mask)
    filtered_image = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(image)) * filter_mask)))
    return np.abs(filtered_image).astype(np.uint8)

def criss_cross_filter(image, frequency):
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    filter_mask = np.sin((x + y / 1.5) * frequency)
    filtered_image = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(image)) * filter_mask)))
    return np.abs(filtered_image).astype(np.uint8)

image = cv2.imread("C:\\DIP\\Dataset\\5.3.01.tiff", cv2.IMREAD_GRAYSCALE)



# Add sine noise
noisy_image = add_sine_noise(image, frequency=0.1)

# Apply band-reject filter
filtered_band_reject = band_reject_filter(noisy_image, frequency=0.1, bandwidth=20)

# Apply criss-cross filter
filtered_criss_cross = criss_cross_filter(noisy_image, frequency=0.1)

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Band-Reject Filter", filtered_band_reject)
cv2.imshow("Criss-Cross Filter", filtered_criss_cross)
cv2.waitKey(0)
cv2.destroyAllWindows()
