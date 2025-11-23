# Hyperspectral Image Processing

Simple implementation of fundamental image processing algorithms from scratch on the Indian Pines hyperspectral dataset.

## Algorithms Implemented

All algorithms are implemented from scratch using only numpy (no pre-written image processing functions):

- **Log Transform** - Enhances dark regions using logarithmic mapping
- **Gamma Correction** - Adjusts image brightness (γ=2.2)
- **Contrast Stretching** - Stretches pixel values between 2nd and 98th percentile
- **Histogram Equalization** - Redistributes pixel intensities for better contrast
- **Mean Filter** - Smooths image using 5x5 averaging kernel
- **Edge Detection** - Detects edges using Sobel gradient filters

## Usage

Place `Indian_pines.mat` in the same folder, then run:

```powershell
C:\Python310\python.exe simple_processing.py
```

## Output

The script generates 6 PNG images in the `outputs/` folder, each showing before and after comparison:

1. `1_log_transform.png`
2. `2_gamma_correction.png`
3. `3_contrast_stretching.png`
4. `4_histogram_equalization.png`
5. `5_mean_filter.png`
6. `6_edge_detection.png`

## Requirements

- numpy
- scipy (for loading .mat file only)
- matplotlib (for visualization only)
