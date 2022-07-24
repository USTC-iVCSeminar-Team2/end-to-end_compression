import numpy as np
from PIL import Image

def peak_signal_noise_ratio(original, compared):
    if isinstance(original, str):
        original = np.array(Image.open(original).convert('RGB'), dtype=np.float32)
    if isinstance(compared, str):
        compared = np.array(Image.open(compared).convert('RGB'), dtype=np.float32)

    mse = np.mean(np.square(original - compared))
    peak_signal_noise_ratio = np.clip(
        np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
    return peak_signal_noise_ratio