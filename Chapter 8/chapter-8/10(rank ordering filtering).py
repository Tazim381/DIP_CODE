import cv2
import numpy as np

# Load the input image
image = cv2.imread("C:\\DIP\\Dataset\\5.3.01.tiff",  cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise with mean 0 and variance 0.01
mean = 0
variance = 0.1
noise = np.random.normal(mean, np.sqrt(variance), image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

# Apply median filtering to remove noise
denoised_rank_order = cv2.medianBlur(noisy_image, ksize=3)

# Display the original, noisy, and denoised images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Denoised (Rank Order Filter)', denoised_rank_order)
cv2.waitKey(0)
cv2.destroyAllWindows()
