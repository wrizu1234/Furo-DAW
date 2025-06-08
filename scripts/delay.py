import numpy as np
import soundfile as sf

def process_delay(input_file, output_file, delay_params):
    mix, delay_ms, spread = delay_params
    feedback = 0.4  # can be another param if you want
    delay_ms = max(1.0, delay_ms * 2000)
    
    # Load audio
    audio, sr = sf.read(input_file)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    num_samples, num_channels = audio.shape
    delay_samples = int(sr * delay_ms / 1000)

    # Output buffer (long enough for echoes)
    output_length = num_samples + delay_samples * 4  # allow 4 echoes
    output = np.zeros((output_length, num_channels), dtype=np.float32)

    # Insert dry signal
    output[:num_samples] += audio * (1 - mix)

    # Insert delayed echoes
    for echo in range(1, 5):  # 4 echoes
        gain = (mix * (feedback ** echo))
        start = delay_samples * echo
        end = start + num_samples
        if end > output_length:
            end = output_length
        segment = audio[:end - start]

        # Apply stereo spread: later delay to R channel
        for ch in range(num_channels):
            ch_start = start + int(spread * delay_samples * ch)
            ch_end = ch_start + segment.shape[0]
            if ch_end > output_length:
                ch_end = output_length
                segment = segment[:ch_end - ch_start]
            output[ch_start:ch_end, ch] += gain * segment[:, ch]

    # Normalize to avoid clipping
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output /= max_val

    # Write output
    sf.write(output_file, output, sr)
