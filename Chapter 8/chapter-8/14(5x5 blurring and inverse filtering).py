import cv2
import numpy as np

def apply_blur(image, kernel_size):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def deblur_inverse_filter(blurred_image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    kernel_freq = np.fft.fft2(kernel, s=blurred_image.shape)
    blurred_image_freq = np.fft.fft2(blurred_image)
    deblurred_image_freq = blurred_image_freq / kernel_freq
    deblurred_image = np.fft.ifft2(deblurred_image_freq)
    deblurred_image = np.abs(deblurred_image).astype(np.uint8)
    return deblurred_image

image_path = "C:\\DIP\\Dataset\\5.3.01.tiff"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply blurring filter
blurred_image = apply_blur(image, kernel_size=5)

# Deblur using inverse filtering
deblurred_image = deblur_inverse_filter(blurred_image, kernel_size=5)

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
cv2.imshow("Deblurred Image", deblurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
