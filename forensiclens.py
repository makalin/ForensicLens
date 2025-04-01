import argparse
import os
from PIL import Image, ExifTags
import imagehash
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tempfile
import shutil
from scipy.fft import fft2, fftshift
from scipy.signal import wiener
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

def analyze_image(file_path):
    print(f"\n=== ForensicLens Analysis: {file_path} ===\n")
    
    if not os.path.exists(file_path):
        print("Error: File does not exist.")
        return

    try:
        image = Image.open(file_path)
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return

    # Basic File Info
    file_size = os.path.getsize(file_path) / 1024
    print(f"File Path: {file_path}")
    print(f"File Size: {file_size:.2f} KB")
    print(f"Image Format: {image.format}")
    print(f"Image Mode: {image.mode}")
    print(f"Image Size: {image.size[0]}x{image.size[1]} pixels")

    # Image Hashing
    print("\n[Image Hashing]")
    phash = imagehash.phash(image)
    print(f"Perceptual Hash: {phash}")

    # Pixel Analysis
    if image.mode == "RGB":
        print("\n[Pixel Analysis]")
        pixels = list(image.getdata())
        r, g, b = 0, 0, 0
        for pixel in pixels:
            r += pixel[0]
            g += pixel[1]
            b += pixel[2]
        pixel_count = len(pixels)
        print(f"Average RGB: ({r//pixel_count}, {g//pixel_count}, {b//pixel_count})")

    # Histogram Analysis
    histogram_analysis(image)

    # Edge Detection
    edge_detection(file_path)

    # Error Level Analysis
    error_level_analysis(file_path)

    # Noise Analysis
    noise_analysis(file_path)

    # Metadata Tampering Detection
    metadata_tampering_detection(image)

    # Frequency Domain Analysis
    frequency_domain_analysis(file_path)

    # Block Artifact Analysis
    block_artifact_analysis(file_path)

    # PRNU Analysis
    prnu_analysis(file_path)

    # ML-Based Tampering Detection
    ml_tampering_detection(file_path)

def histogram_analysis(image):
    print("\n[Histogram Analysis]")
    img = image.convert("RGB")
    colors = ("Red", "Green", "Blue")
    for i, color in enumerate(colors):
        hist = np.histogram(np.array(img)[:, :, i], bins=256, range=(0, 256))[0]
        mean = np.mean(hist)
        std_dev = np.std(hist)
        print(f"{color} Channel - Mean: {mean:.2f}, Std Dev: {std_dev:.2f}")
    plt.figure(figsize=(6, 4))
    for i, color in enumerate(['r', 'g', 'b']):
        plt.hist(np.array(img)[:, :, i].ravel(), bins=256, color=color, alpha=0.5, label=f"{colors[i]}")
    plt.legend()
    plt.title("RGB Histogram")
    plt.savefig("histogram.png")
    print("Histogram saved as 'histogram.png'")

def edge_detection(file_path):
    print("\n[Edge Detection (Canny)]")
    img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        print("Failed to load image.")
        return
    edges = cv2.Canny(img_cv, 100, 200)
    edge_count = np.sum(edges > 0)
    total_pixels = edges.size
    edge_ratio = (edge_count / total_pixels) * 100
    print(f"Edge Pixels: {edge_count} ({edge_ratio:.2f}% of total)")
    cv2.imwrite("edges.png", edges)
    print("Edge-detected image saved as 'edges.png'")

def error_level_analysis(file_path):
    print("\n[Error Level Analysis (ELA)]")
    try:
        img = Image.open(file_path).convert("RGB")
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp.jpg")
        img.save(temp_path, "JPEG", quality=95)
        img_recompressed = Image.open(temp_path).convert("RGB")
        img_array = np.array(img)
        recomp_array = np.array(img_recompressed)
        diff = np.abs(img_array - recomp_array).astype(np.uint8)
        ela = np.mean(diff) * 10
        ela_img = Image.fromarray(diff)
        ela_img.save("ela.png")
        print(f"ELA Mean Difference: {ela:.2f}")
        print("ELA image saved as 'ela.png'")
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"ELA Error: {str(e)}")

def noise_analysis(file_path):
    print("\n[Noise Analysis]")
    img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        print("Failed to load image.")
        return
    blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
    noise = cv2.absdiff(img_cv, blurred)
    noise_mean = np.mean(noise)
    noise_std = np.std(noise)
    print(f"Noise Mean: {noise_mean:.2f}, Noise Std Dev: {noise_std:.2f}")
    cv2.imwrite("noise.png", noise * 10)
    print("Noise map saved as 'noise.png'")

def metadata_tampering_detection(image):
    print("\n[Metadata Tampering Detection]")
    try:
        exif_data = image._getexif()
        if not exif_data:
            print("No EXIF data found.")
            return
        suspicious = False
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "DateTime" and isinstance(value, str):
                try:
                    from datetime import datetime
                    img_date = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    if img_date > datetime(2025, 3, 31):
                        suspicious = True
                        print(f"Suspicious DateTime: {value} (future date)")
                except ValueError:
                    suspicious = True
                    print(f"Invalid DateTime format: {value}")
            print(f"{tag}: {value}")
        if suspicious:
            print("Warning: Possible metadata tampering detected.")
    except AttributeError:
        print("No EXIF data available.")

def frequency_domain_analysis(file_path):
    print("\n[Frequency Domain Analysis (FFT)]")
    img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        print("Failed to load image.")
        return
    f = fft2(img_cv)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    mean_magnitude = np.mean(magnitude_spectrum)
    std_magnitude = np.std(magnitude_spectrum)
    print(f"Mean Magnitude: {mean_magnitude:.2f}, Std Dev: {std_magnitude:.2f}")
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("Frequency Spectrum (FFT)")
    plt.savefig("fft_spectrum.png")
    print("Frequency spectrum saved as 'fft_spectrum.png'")

def block_artifact_analysis(file_path):
    print("\n[Block Artifact Analysis]")
    img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        print("Failed to load image.")
        return
    grad_x = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.SobelRisks    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    blockiness = np.var(grad_mag) / (grad_mag.shape[0] * grad_mag.shape[1] / 64)  # 8x8 blocks
    print(f"Blockiness Score: {blockiness:.2f}")
    grad_mag_normalized = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("block_artifacts.png", grad_mag_normalized)
    print("Block artifact map saved as 'block_artifacts.png'")

def prnu_analysis(file_path):
    print("\n[PRNU Analysis]")
    img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        print("Failed to load image.")
        return
    
    # Estimate noise pattern (simplified PRNU)
    blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
    noise = img_cv.astype(float) - blurred.astype(float)
    prnu = wiener(noise, mysize=(5, 5))  # Wiener filter to enhance noise pattern
    prnu_mean = np.mean(prnu)
    prnu_std = np.std(prnu)
    print(f"PRNU Mean: {prnu_mean:.2f}, Std Dev: {prnu_std:.2f}")
    print("Note: PRNU can be compared across images for source identification.")
    
    # Save PRNU pattern
    prnu_normalized = cv2.normalize(prnu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("prnu.png", prnu_normalized)
    print("PRNU pattern saved as 'prnu.png'")

def ml_tampering_detection(file_path):
    print("\n[ML-Based Tampering Detection]")
    img_cv = cv2.imread(file_path)
    if img_cv is None:
        print("Failed to load image.")
        return
    
    # Simple feature extraction (ELA + edge features)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    ela = error_level_analysis_features(file_path)
    features = np.array([np.mean(ela), np.std(ela), np.mean(edges), np.std(edges)])
    
    # Load or train a simple SVM model (placeholder - requires training data)
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("svm_model.pkl")
    except FileNotFoundError:
        print("Warning: ML model not found. Training a placeholder model (requires data).")
        scaler = StandardScaler()
        model = SVC(probability=True)
        # Placeholder training (replace with real data)
        X_train = np.random.rand(10, 4)  # Dummy data
        y_train = np.random.randint(0, 2, 10)
        scaler.fit(X_train)
        model.fit(scaler.transform(X_train), y_train)
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(model, "svm_model.pkl")
    
    # Predict tampering probability
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = model.predict_proba(features_scaled)[0][1]
    print(f"Tampering Probability: {prob:.2%}")
    print("Note: High probability suggests potential tampering (model accuracy depends on training).")

def error_level_analysis_features(file_path):
    # Helper function for ML feature extraction
    img = Image.open(file_path).convert("RGB")
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp.jpg")
    img.save(temp_path, "JPEG", quality=95)
    img_recompressed = Image.open(temp_path).convert("RGB")
    diff = np.abs(np.array(img) - np.array(img_recompressed)).astype(np.uint8)
    shutil.rmtree(temp_dir)
    return diff

def main():
    parser = argparse.ArgumentParser(description="ForensicLens: A terminal-based forensic image analyzer.")
    parser.add_argument("image_path", help="Path to the image file to analyze")
    args = parser.parse_args()
    analyze_image(args.image_path)

if __name__ == "__main__":
    main()
    