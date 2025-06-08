import numpy as np
from scipy.io import wavfile

def apply_volume(input_path, output_path, volume):
    """
    Adjusts the volume of a WAV audio file.
    
    Parameters:
        input_path (str): Path to the input .wav file.
        output_path (str): Path to save the output .wav file.
        volume (float): Volume multiplier (e.g., 0.5 = half, 2.0 = double).
    """
    # Read the WAV file
    sample_rate, data = wavfile.read(input_path)

    # Ensure data is in float32 for scaling, handle int types
    if data.dtype == np.int16:
        max_val = np.iinfo(np.int16).max
        data = data.astype(np.float32) / max_val
    elif data.dtype == np.int32:
        max_val = np.iinfo(np.int32).max
        data = data.astype(np.float32) / max_val
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        raise TypeError(f"Unsupported audio data type: {data.dtype}")

    # Apply volume change
    data = data * volume

    # Clip to prevent overflow
    data = np.clip(data, -1.0, 1.0)

    # Convert back to int16 and save
    output_data = (data * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, output_data)

def apply_pan(input_path, output_path, pan):
    """
    Sets the stereo pan of a WAV audio file.
    
    Parameters:
        input_path (str): Path to input .wav file.
        output_path (str): Path to save output .wav file.
        pan (float): Pan value (0 = left, 1 = right, 0.5 = center).
    """
    if not (0 <= pan <= 1):
        raise ValueError("Pan must be between 0 and 1.")

    sample_rate, data = wavfile.read(input_path)

    # Normalize to float
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        raise TypeError("Only 16-bit PCM WAV files are supported.")

    # If mono, duplicate to stereo
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)

    # Compute pan multipliers using equal power panning
    left_mult = np.cos(pan * np.pi / 2)
    right_mult = np.sin(pan * np.pi / 2)

    # Apply pan
    data[:, 0] *= left_mult
    data[:, 1] *= right_mult

    # Clip and convert back
    data = np.clip(data, -1.0, 1.0)
    output_data = (data * 32767).astype(np.int16)

    wavfile.write(output_path, sample_rate, output_data)

apply_pan("input.wav", "out.wav", 1)