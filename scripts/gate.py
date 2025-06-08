from pydub import AudioSegment
from pydub.utils import make_chunks

def lerp(t, start, end):
    return (1 - t) * start + t * end

def apply_gain_and_noise_gate(input_path, output_path, xx, chunk_ms=10):
    gain_db, threshold_db = xx
    gain_db = lerp(gain_db, -60, 60)
    threshold_db = lerp(threshold_db, -60, 0)

    audio = AudioSegment.from_file(input_path)
    chunks = make_chunks(audio, chunk_ms)
    processed_chunks = []
    for chunk in chunks:
        # Apply gain
        chunk = chunk.apply_gain(gain_db)
        # If chunk's dBFS is below threshold, silence it
        if chunk.dBFS < threshold_db:
            chunk = AudioSegment.silent(duration=len(chunk), frame_rate=audio.frame_rate)
        processed_chunks.append(chunk)
    output = sum(processed_chunks)
    output.export(output_path, format="wav")

