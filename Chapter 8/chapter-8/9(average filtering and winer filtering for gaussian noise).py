import cv2
import numpy as np

def denoise_average(image, kernel_size):
    # Apply average filtering to the image using a square kernel
    blurred_image = cv2.blur(image, (kernel_size, kernel_size))
   
    return blurred_image

def denoise_wiener(image, kernel_size, noise_variance):
    # Estimate the noise power spectrum
    noise_power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2
   
    # Estimate the signal power spectrum
    signal_power_spectrum = np.mean(noise_power_spectrum)
   
    # Estimate the Wiener filter transfer function
    wiener_filter = np.conj(noise_power_spectrum) / (noise_power_spectrum + noise_variance * signal_power_spectrum)
   
    # Apply the Wiener filter to the image in the frequency domain
    filtered_image = np.real(np.fft.ifft2(np.fft.ifftshift(wiener_filter * np.fft.fftshift(np.fft.fft2(image)))))
   
    # Clip the pixel values to the valid range of 0-255
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
   
    return filtered_image

# Load the input image
image = cv2.imread("C:\\DIP\\Dataset\\5.3.01.tiff", cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise with mean 0 and variance 0.01
mean = 0
variance = 0.01
noise = np.random.normal(mean, np.sqrt(variance), image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

# Apply average filtering to remove noise
denoised_average = denoise_average(noisy_image, kernel_size=3)

# Apply Wiener filtering to remove noise
denoised_wiener = denoise_wiener(noisy_image, kernel_size=3, noise_variance=0.01)

# Display the original, noisy, and denoised images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Denoised (Average Filter)', denoised_average)
cv2.imshow('Denoised (Wiener Filter)', denoised_wiener)
cv2.waitKey(0)
cv2.destroyAllWindows()
