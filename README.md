# ForensicLens

![ForensicLens logo](forensiclens_logo.png)

**ForensicLens** is a powerful, terminal-based forensic image analysis tool designed to detect tampering, verify integrity, and extract detailed insights from digital images. Built with Python, it leverages advanced techniques like PRNU analysis, machine learning, and frequency domain analysis to assist forensic investigators, researchers, and enthusiasts.

## Features

- **Basic Metadata Extraction**: File size, format, dimensions, and mode.
- **Perceptual Hashing**: Generates a hash for integrity checking and similarity comparison.
- **Pixel Analysis**: Computes average RGB values for basic color profiling.
- **Histogram Analysis**: Plots RGB histograms to detect anomalies (saved as `histogram.png`).
- **Edge Detection**: Applies Canny edge detection to highlight potential edits (saved as `edges.png`).
- **Error Level Analysis (ELA)**: Identifies compression inconsistencies (saved as `ela.png`).
- **Noise Analysis**: Estimates noise levels to spot tampering (saved as `noise.png`).
- **Metadata Tampering Detection**: Checks EXIF data for inconsistencies (e.g., future dates).
- **Frequency Domain Analysis (FFT)**: Detects periodic noise or resampling artifacts (saved as `fft_spectrum.png`).
- **Block Artifact Analysis**: Identifies JPEG compression blocks that may indicate edits (saved as `block_artifacts.png`).
- **PRNU Analysis**: Extracts camera-specific noise patterns for source identification (saved as `prnu.png`).
- **Machine Learning-Based Tampering Detection**: Uses an SVM classifier to predict tampering probability (requires training).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/makalin/ForensicLens.git
   cd ForensicLens
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.x installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install:
   ```bash
   pip install Pillow imagehash numpy opencv-python matplotlib scipy scikit-learn joblib
   ```

3. **Verify Setup**:
   Test with a sample image:
   ```bash
   python forensiclens.py sample.jpg
   ```

## Requirements

- **Python**: 3.6 or higher
- **Libraries**: 
  - `Pillow`: Image processing
  - `imagehash`: Perceptual hashing
  - `numpy`: Numerical computations
  - `opencv-python`: Image analysis (e.g., edge detection)
  - `matplotlib`: Plotting (e.g., histograms)
  - `scipy`: Signal processing (e.g., FFT, Wiener filter)
  - `scikit-learn`: Machine learning (SVM)
  - `joblib`: Model persistence

Create a `requirements.txt` file with:
```
Pillow
imagehash
numpy
opencv-python
matplotlib
scipy
scikit-learn
joblib
```

## Usage

Run the tool from the terminal by providing an image path:
```bash
python forensiclens.py path/to/image.jpg
```

### Example Output
```
=== ForensicLens Analysis: test.jpg ===

File Path: test.jpg
File Size: 123.45 KB
Image Format: JPEG
Image Mode: RGB
Image Size: 1920x1080 pixels

[Image Hashing]
Perceptual Hash: 9f8c7e3d1b2a4f6c

[Pixel Analysis]
Average RGB: (120, 134, 98)

[Histogram Analysis]
Red Channel - Mean: 123.45, Std Dev: 67.89
Histogram saved as 'histogram.png'

[PRNU Analysis]
PRNU Mean: 0.12, Std Dev: 5.67
PRNU pattern saved as 'prnu.png'

[ML-Based Tampering Detection]
Tampering Probability: 73.45%
```

### Output Files
- `histogram.png`: RGB histogram plot
- `edges.png`: Canny edge detection result
- `ela.png`: Error Level Analysis difference map
- `noise.png`: Noise map (amplified)
- `fft_spectrum.png`: Frequency domain spectrum
- `block_artifacts.png`: Block artifact gradient map
- `prnu.png`: PRNU noise pattern

## Notes

- **PRNU Analysis**: This is a simplified implementation. For accurate camera source identification, calibrate with multiple images from the same device.
- **ML Tampering Detection**: The included SVM model is a placeholder. For production use, train it with a labeled dataset (e.g., CASIA Tampered Image Detection Dataset) and update `scaler.pkl` and `svm_model.pkl`.
- **Performance**: Large images may take longer to process, especially for FFT and ML features.

## Contributing

Contributions are welcome! Here’s how to get started:
1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to your fork: `git push origin feature/your-feature`.
5. Open a Pull Request.

### Ideas for Enhancement
- Add support for video frame analysis.
- Implement additional ML models (e.g., CNNs with PyTorch/TensorFlow).
- Optimize for batch processing of multiple images.
- Integrate a GUI option alongside the terminal interface.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with inspiration from forensic image analysis research and open-source tools.
- Special thanks to the Python community and libraries like OpenCV and scikit-learn.
