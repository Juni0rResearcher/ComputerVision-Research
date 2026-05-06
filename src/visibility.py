from enum import Enum

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import hog


class VisibilityCondition(Enum):
    NIGHT = "night"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    CLEAR = "clear"


def compute_hog_features(gray: np.ndarray) -> np.ndarray:
    return hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)


def detect_visibility_condition(image: np.ndarray) -> VisibilityCondition:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    contrast = gray.std()
    high_freq_energy = cv2.Laplacian(gray, cv2.CV_64F).var()
    hog_sparsity = np.mean(compute_hog_features(gray) < 0.01)
    horizontal_edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    vertical_edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    if mean_brightness < 60:
        return VisibilityCondition.NIGHT
    if contrast < 40 and hog_sparsity > 0.7:
        return VisibilityCondition.FOG
    if high_freq_energy > 800:
        return VisibilityCondition.RAIN if vertical_edges.var() > horizontal_edges.var() * 1.5 else VisibilityCondition.SNOW
    return VisibilityCondition.CLEAR


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    mean_guide = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
    mean_src = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
    corr_guide = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
    corr_guide_src = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))
    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_src = corr_guide_src - mean_guide * mean_src
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    return mean_a * guide + mean_b


def enhance_night(image: np.ndarray) -> np.ndarray:
    mean_rgb = image.mean(axis=(0, 1))
    gray_value = mean_rgb.mean()
    scale = gray_value / mean_rgb
    image = np.clip(image * scale, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_dark = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
    l_bright = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    mask = (l < 50).astype(np.float32)
    l_enhanced = (l_dark * mask + l_bright * (1 - mask)).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2RGB)


def enhance_rain(image: np.ndarray) -> np.ndarray:
    denoised = cv2.medianBlur(image, 5)
    enhanced = cv2.addWeighted(image, 0.7, denoised, 0.3, 0)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def enhance_snow(image: np.ndarray) -> np.ndarray:
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=7)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2RGB)


def enhance_fog_dcp(image: np.ndarray, omega: float = 0.85, t0: float = 0.1) -> np.ndarray:
    img_float = image.astype(np.float64) / 255.0
    dark_channel = ndi.minimum_filter(img_float.min(axis=2), size=15)

    num_pixels = max(1, dark_channel.size // 1000)
    flat_img = img_float.reshape(-1, 3)
    flat_dark = dark_channel.reshape(-1)
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    atmospheric_light = np.max(flat_img[indices], axis=0)

    transmission = 1 - omega * dark_channel
    transmission = guided_filter(img_float.mean(axis=2), transmission, radius=40, eps=1e-6)
    transmission = np.maximum(transmission, t0)

    enhanced = np.zeros_like(img_float)
    for c in range(3):
        enhanced[:, :, c] = (img_float[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]

    return (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)


def preprocess_for_visibility(image: np.ndarray, condition: VisibilityCondition | None = None) -> np.ndarray:
    if condition is None:
        condition = detect_visibility_condition(image)
    if condition == VisibilityCondition.NIGHT:
        return enhance_night(image)
    if condition == VisibilityCondition.RAIN:
        return enhance_rain(image)
    if condition == VisibilityCondition.SNOW:
        return enhance_snow(image)
    if condition == VisibilityCondition.FOG:
        return enhance_fog_dcp(image)
    return image
