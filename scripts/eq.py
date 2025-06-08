import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.999)
    return butter(order, [low, high], btype='band')

def apply_bandpass(data, lowcut, highcut, fs, gain, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = lfilter(b, a, data)
    return filtered * gain

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return np.zeros_like(audio)
    return audio / max_val

def process_wav_file(input_file, output_file, gains, sample_rate=44100, normalize=False):
    if len(gains) != 7:
        raise ValueError("Gains must be a list of 7 values between 0 and 1.")
    
    # Load audio
    audio = AudioSegment.from_file(input_file).set_channels(1).set_frame_rate(sample_rate)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Normalize to [-1.0, 1.0]
    samples /= np.iinfo(audio.array_type).max

    output = np.zeros_like(samples)
    for i, gain in enumerate(gains):
        center_freq = (2 ** i) * 100
        band_low = center_freq / np.sqrt(2)
        band_high = center_freq * np.sqrt(2)

        if gain > 0:
            if gain == 0.5:
                scale = 1.0  # Neutral gain
            elif gain == 1.0:
                scale = 2.0  # Boosted
            else:
                scale = gain * 2  # Scale between 0 and 2
            band = apply_bandpass(samples, band_low, band_high, sample_rate, gain=scale)
            output += band  # Accumulate

    # Optional normalization
    if normalize:
        output = normalize_audio(output)

    # Clip to [-1.0, 1.0] and convert back to original format
    output = np.clip(output, -1.0, 1.0)
    output_int = (output * np.iinfo(audio.array_type).max).astype(audio.array_type)

    # Save to output
    output_audio = audio._spawn(output_int.tobytes())
    output_audio.export(output_file, format="wav")
