from pedalboard import Pedalboard, Chorus
from pedalboard.io import AudioFile

def lerp(t, start, end):
    return (1 - t) * start + t * end

def apply_chorus(input_path, output_path, params=None):
    """
    Apply a chorus effect to an audio file using a list of parameters.
    
    Parameters:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the processed audio file.
        params (list): [rate_hz, depth, centre_delay_ms, feedback, mix]
                       All elements must be floats.
    """
    # Default chorus if no params given

    rate_hz, depth, centre_delay_ms, feedback, mix = params
    rate_hz = lerp(rate_hz, 0.1, 5)
    centre_delay_ms = lerp(feedback, 1, 30)
    feedback = lerp(feedback, -0.95, 0.95)

    # Create the chorus effect
    board = Pedalboard([
        Chorus(
            rate_hz=rate_hz,
            depth=depth,
            centre_delay_ms=centre_delay_ms,
            feedback=feedback,
            mix=mix
        )
    ])

    # Read input file
    with AudioFile(input_path, 'r') as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate

    # Apply effect
    effected = board(audio, samplerate)

    # Write output file
    with AudioFile(output_path, 'w', samplerate, effected.shape[0]) as f:
        f.write(effected)
