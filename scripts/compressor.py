import numpy as np
import soundfile as sf

def lerp(t, start, end):
    return (1 - t) * start + t * end

def output(x): print(x)

def apply_compression(input_path, output_path, knob_values):
    threshold_db, ratio, attack_ms, release_ms = knob_values
    threshold_db = lerp(threshold_db, -60, 0)
    ratio = lerp(ratio, 1, 20)
    attack_ms = lerp(attack_ms, 0.1, 200)
    release_ms = lerp(release_ms, 10, 2000)
    audio, sr = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono

    threshold = 10 ** (threshold_db / 20)
    attack = np.exp(-1 / (sr * attack_ms / 1000))
    release = np.exp(-1 / (sr * release_ms / 1000))

    gain = 1.0
    output = np.zeros_like(audio)

    for i in range(len(audio)):
        level = abs(audio[i])
        if level > threshold:
            target_gain = (threshold + (level - threshold) / ratio) / level
        else:
            target_gain = 1.0

        if target_gain < gain:
            gain = attack * gain + (1 - attack) * target_gain
        else:
            gain = release * gain + (1 - release) * target_gain

        output[i] = audio[i] * gain

    sf.write(output_path, output, sr)


