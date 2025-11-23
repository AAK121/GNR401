import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

def load_data(path):
    data = loadmat(path)
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v.astype(float)
    return None

def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val == min_val:
        return img
    return (img - min_val) / (max_val - min_val)

def log_transform(img):
    normalized = normalize(img)
    result = np.log(1 + normalized)
    return normalize(result)

def gamma_correction(img, gamma=2.2):
    normalized = normalize(img)
    result = np.power(normalized, 1.0/gamma)
    return result

def contrast_stretching(img):
    flat = img.flatten()
    flat_sorted = np.sort(flat)
    n = len(flat_sorted)
    low_idx = int(0.02 * n)
    high_idx = int(0.98 * n)
    low_val = flat_sorted[low_idx]
    high_val = flat_sorted[high_idx]
    
    result = (img - low_val) / (high_val - low_val)
    result = np.clip(result, 0, 1)
    return result

def histogram_equalization(img):
    normalized = normalize(img)
    img_int = (normalized * 255).astype(int)
    
    hist = np.zeros(256)
    for val in img_int.flatten():
        hist[val] += 1
    
    cdf = np.zeros(256)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    
    cdf = cdf / cdf[-1]
    
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = cdf[img_int[i, j]]
    
    return result

def convolve2d(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(patch * kernel)
    
    return result

def mean_filter(img, size=3):
    kernel = np.ones((size, size)) / (size * size)
    return convolve2d(img, kernel)

def gradient_filter(img):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=float)
    
    gx = convolve2d(img, sobel_x)
    gy = convolve2d(img, sobel_y)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    return normalize(magnitude)

cube = load_data('Indian_pines.mat')
print(f"Loaded data shape: {cube.shape}")

band = cube[:, :, 50]
band = normalize(band)

os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(band, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

log_result = log_transform(band)
axes[1].imshow(log_result, cmap='gray')
axes[1].set_title('Log Transform')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('outputs/1_log_transform.png')
plt.close()
print("Saved: 1_log_transform.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(band, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

gamma_result = gamma_correction(band, gamma=2.2)
axes[1].imshow(gamma_result, cmap='gray')
axes[1].set_title('Gamma Correction (γ=2.2)')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('outputs/2_gamma_correction.png')
plt.close()
print("Saved: 2_gamma_correction.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(band, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

contrast_result = contrast_stretching(band)
axes[1].imshow(contrast_result, cmap='gray')
axes[1].set_title('Contrast Stretching')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('outputs/3_contrast_stretching.png')
plt.close()
print("Saved: 3_contrast_stretching.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(band, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

histeq_result = histogram_equalization(band)
axes[1].imshow(histeq_result, cmap='gray')
axes[1].set_title('Histogram Equalization')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('outputs/4_histogram_equalization.png')
plt.close()
print("Saved: 4_histogram_equalization.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(band, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

mean_result = mean_filter(band, size=5)
axes[1].imshow(mean_result, cmap='gray')
axes[1].set_title('Mean Filter (5x5)')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('outputs/5_mean_filter.png')
plt.close()
print("Saved: 5_mean_filter.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(band, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

edge_result = gradient_filter(band)
axes[1].imshow(edge_result, cmap='gray')
axes[1].set_title('Edge Detection (Sobel)')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('outputs/6_edge_detection.png')
plt.close()
print("Saved: 6_edge_detection.png")

print("\nAll processing complete!")
