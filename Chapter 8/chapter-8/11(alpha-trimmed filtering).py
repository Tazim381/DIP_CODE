import numpy as np
import cv2

def add_gaussian_noise(image, mean, variance):
    noise = np.random.normal(mean, np.sqrt(variance), image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def alpha_trimmed_mean_filter(image, d, alpha):
    filtered_image = np.zeros_like(image)
    pad_width = d // 2
    padded_image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width)), mode='reflect')
   
    for i in range(pad_width, padded_image.shape[0] - pad_width):
        for j in range(pad_width, padded_image.shape[1] - pad_width):
            neighborhood = padded_image[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]
            sorted_values = np.sort(neighborhood.flatten())
            trimmed_values = sorted_values[alpha:d**2-alpha]
            filtered_image[i-pad_width, j-pad_width] = np.mean(trimmed_values)
   
    return filtered_image.astype(np.uint8)

def geometric_mean_filter(image, d):
    filtered_image = np.zeros_like(image)
    pad_width = d // 2
    padded_image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width)), mode='reflect')
   
    for i in range(pad_width, padded_image.shape[0] - pad_width):
        for j in range(pad_width, padded_image.shape[1] - pad_width):
            neighborhood = padded_image[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]
            filtered_image[i-pad_width, j-pad_width] = np.exp(np.mean(np.log(neighborhood)))
   
    return filtered_image.astype(np.uint8)

image = cv2.imread("C:\\DIP\\Dataset\\5.3.01.tiff", cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noisy_image = add_gaussian_noise(image, 0, 0.1)

# Apply alpha-trimmed mean filter
filtered_alpha_trimmed = alpha_trimmed_mean_filter(noisy_image, d=3, alpha=1)

# Apply geometric mean filter
filtered_geometric_mean = geometric_mean_filter(noisy_image, d=3)

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Alpha-Trimmed Mean Filter", filtered_alpha_trimmed)
cv2.imshow("Geometric Mean Filter", filtered_geometric_mean)
cv2.waitKey(0)
cv2.destroyAllWindows()
