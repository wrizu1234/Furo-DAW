from pydub import AudioSegment
import imageio_ffmpeg
import os

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
TRACK_HEIGHT = 60
MENU_OFFSET = 120
def render(tracks, lps, acc, x_off, automations, seq):
    tracks = tracks.copy()
    max_time = 0
    for track in tracks:
        track.TIME_x = track.rect.x + (acc - x_off)
        clip_end = (track.TIME_x + track.rect.width) / (lps * (seq.tempo / 240))
        if clip_end > max_time:
            max_time = clip_end

    print(f"Total duration: {max_time:.2f} seconds")
    master = AudioSegment.silent(duration=int(max_time * 1000))  # in ms

    for track_idx, track in enumerate(tracks):
        start_time = int((track.TIME_x / (lps * (seq.tempo / 240))) * 1000)
        ext = os.path.splitext(track.path)[1][1:]  # Get extension without dot
        segment = AudioSegment.from_file(track.path, format=ext)
        dur = track.rect.width / (lps * (seq.tempo / 240))
        segment = segment[:int(dur * 1000)]

        # Apply volume automation if it exists
        if len(automations[track_idx]["volume"]) > 0:
            # Get automation points for this track
            points = automations[track_idx]["volume"]
            points.sort(key=lambda p: p.pos[0])  # Sort by x position

            # Split segment into chunks and apply volume changes
            chunk_size = 10  # Process in 10ms chunks
            processed_segments = []
            
            for chunk_start in range(0, len(segment), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(segment))
                chunk = segment[chunk_start:chunk_end]
                
                # Calculate average volume for this chunk
                chunk_volumes = []
                for ms in range(chunk_start, chunk_end):
                    current_x = track.TIME_x + (ms / 1000 * lps * (seq.tempo / 240))  # Convert ms to x position
                    
                    # Find surrounding points
                    before_point = None
                    after_point = None
                    for point in points:
                        if point.pos[0] <= current_x:
                            before_point = point
                        else:
                            after_point = point
                            break

                    # Calculate volume using interpolation
                    if before_point and after_point:
                        x1, y1 = before_point.pos
                        x2, y2 = after_point.pos
                        vol1 = 1 - (y1 - (MENU_OFFSET + track_idx*TRACK_HEIGHT)) / TRACK_HEIGHT
                        vol2 = 1 - (y2 - (MENU_OFFSET + track_idx*TRACK_HEIGHT)) / TRACK_HEIGHT
                        t = (current_x - x1) / (x2 - x1)
                        vol = vol1 + (vol2 - vol1) * t
                    elif before_point:
                        y = before_point.pos[1]
                        vol = 1 - (y - (MENU_OFFSET + track_idx*TRACK_HEIGHT)) / TRACK_HEIGHT
                    elif after_point:
                        y = after_point.pos[1]
                        vol = 1 - (y - (MENU_OFFSET + track_idx*TRACK_HEIGHT)) / TRACK_HEIGHT
                    else:
                        vol = 1.0

                    # Clamp volume between 0 and 1
                    vol = max(0, min(1, vol))
                    chunk_volumes.append(vol)
                
                # Apply average volume to chunk
                avg_vol = sum(chunk_volumes) / len(chunk_volumes)
                chunk = chunk - (1 - avg_vol) * 60  # Convert to dB
                processed_segments.append(chunk)
            
            # Combine processed chunks
            segment = sum(processed_segments)

        master = master.overlay(segment, position=start_time)

    output_path = "mixdown.wav"
    master.export(output_path, format="wav")
    print(f"Mixdown saved to {output_path}")


import os
import shutil
import tkinter as tk
from tkinter import filedialog
def dnld():
    source_file = 'mixdown.wav'  # or a full path like '/path/to/example.wav'
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    destination = os.path.join(downloads_folder, os.path.basename(source_file))
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.asksaveasfilename(
        defaultextension=".wav",
        filetypes=[("Wav files", "*.wav"), ("All files", "*.*")]
    )

    if file_path:
        shutil.copy(source_file, file_path)

        print(f"file saved at: {file_path}")
    else:
        print("no file selected.")

