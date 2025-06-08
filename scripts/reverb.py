from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
import numpy as np

def process_reverb(input_file, output_file, reverb_params, tail_seconds=4.0, pre_delay_seconds=0.2):
    reverb_size, reverb_mix, reverb_tone = reverb_params

    # Clamp values
    reverb_size = max(0.0, min(1.0, reverb_size))
    reverb_mix = max(0.0, min(1.0, reverb_mix))
    reverb_tone = max(0.0, min(1.0, reverb_tone))
    pre_delay_seconds = max(0.0, pre_delay_seconds)

    # Read input
    with AudioFile(input_file, 'r') as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    num_channels = audio.shape[0] if audio.ndim > 1 else 1
    total_samples = audio.shape[-1]
    tail_samples = int(sample_rate * tail_seconds)
    pre_delay_samples = int(sample_rate * pre_delay_seconds)

    # Pad audio for reverb tail
    if audio.ndim == 1:
        padded_audio = np.pad(audio, (0, tail_samples))
    else:
        padded_audio = np.pad(audio, ((0, 0), (0, tail_samples)), mode='constant')

    # Create delayed version for reverb input
    if audio.ndim == 1:
        delayed_audio = np.pad(audio, (pre_delay_samples, tail_samples))
    else:
        delayed_audio = np.pad(audio, ((0, 0), (pre_delay_samples, tail_samples)), mode='constant')

    # Apply reverb to delayed audio (wet only)
    board = Pedalboard([
        Reverb(
            room_size=reverb_size,
            damping=1.0 - reverb_tone,
            wet_level=1.0,
            dry_level=0.0,
            width=1.0,
            freeze_mode=0.0
        )
    ])
    wet_signal = board(delayed_audio, sample_rate)

    # Ensure both signals are same length before mixing
    min_len = min(padded_audio.shape[-1], wet_signal.shape[-1])
    padded_audio = padded_audio[:, :min_len]
    wet_signal = wet_signal[:, :min_len]

    # Mix
    output = (1.0 - reverb_mix) * padded_audio + reverb_mix * wet_signal

    # Save
    with AudioFile(output_file, 'w', sample_rate, output.shape[0]) as f:
        f.write(output)
