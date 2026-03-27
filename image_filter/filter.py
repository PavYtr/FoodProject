import cv2
import numpy as np

BLUR_THRESHOLD = 80
NOISE_THRESHOLD = 15.0
BRIGHTNESS_LOW = 40
BRIGHTNESS_HIGH = 215


def analyze_image(image_path):
    image = cv2.imread(str(image_path))

    defects = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        defects.append(f"Blurry({int(laplacian_var)})")

    # noise
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    sigma = np.sum(np.abs(cv2.filter2D(gray, -1, M)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    if sigma > NOISE_THRESHOLD:
        defects.append(f"Noisy({sigma:.1f})")

    # brightness
    brightness = np.mean(gray)
    if brightness < BRIGHTNESS_LOW:
        defects.append(f"TooDark({int(brightness)})")
    elif brightness > BRIGHTNESS_HIGH:
        defects.append(f"TooBright({int(brightness)})")

    return defects if defects else None