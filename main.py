import pygame
import bisect
import math
import scripts.renderer as renderer
import os
import queue
import shutil
import time
import librosa
import uuid
import numpy as np
import sounddevice as sd
import tkinter as tk
import soundfile as sf
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.signal import resample
from threading import Thread
from tkinter import filedialog
from pedalboard import *
from pedalboard.io import *
from scripts.visualizers import *
from scripts.eq import *
from scripts.reverb import *
from scripts.delay import *
from scripts.compressor import *
from scripts.gate import *
from scripts.chorus import *
from scripts.mixer_essentials import *

root = tk.Tk()
root.withdraw()
pygame.init()

CHANNELS = 8

samplerate = 44100
dtype = 'int16'
q = queue.Queue()
audio_frames = []
def audio_callback(indata, frames, time, status):
    if recording:
        q.put(indata.copy())

def start_stream():
    global stream
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, dtype=dtype)
    stream.start()

def Image(p):
    return pygame.image.load(p).convert_alpha()
    
# Constants
WIDTH, HEIGHT = 1000, 600
FPS = 60

X_OFF = 150
MENU_OFFSET = 120
TRACK_HEIGHT = 60
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 30

# length per second
LPS = 80

MAXWIDTH = 500*LPS

WHITE = (255,255,255)
BLACK = (0,0,0)
BUTTON_COL = (75,75,75)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Setup window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Plasmatic's DAW")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 22)
font10 = pygame.font.SysFont(None, 15)
font35 = pygame.font.SysFont(None, 35)


def draw_text(surface, text, font, position, color=(255, 255, 255)):
    text_surface = font.render(str(text), True, color)
    surface.blit(text_surface, position)

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def time_format(time):
    if len(str(time)) == 1:
        return f"0{time}"
    elif len(str(time)) == 0:
        return "00"
    else:
        return time

RS = {}

def semi_formula(x):
    return 12 * math.log2(1/x)

def set_pan(channel: pygame.mixer.Channel, pan):
    # Clamp pan between -1.0 and 1.0
    pan = max(-1.0, min(1.0, pan))
    left = 1.0 - max(0.0, pan)
    right = 1.0 + min(0.0, pan)
    channel.set_volume(left, right)

chosen_effect = "volume"

# Track and clip management
tracks = []
audio_clips = []
playing = False
playback_x = 0
start_time = 0
big_ahh_cache = {}
def draw_waveform(filename, surface, color=(0, 255, 0), position=(0, 0), size=(1000, 200)):
    if (filename, color, position, size) not in big_ahh_cache:
        try:
            rate, data = wavfile.read(filename)
        except Exception as e:
            return

        # Handle stereo by taking one channel
        if data.ndim > 1:
            data = data[:, 0]

        # Normalize the audio data
        data = data / np.max(np.abs(data))
        # Downsample if needed
        width, height = size
        samples_per_pixel = max(1, len(data) // width)
        scaled_data = data[::samples_per_pixel][:width]

        # Convert to screen coordinates
        midline = position[1] + height // 2
        amplitude = (height // 2) - 5
        points = [
        ]
        for x, value in enumerate(scaled_data):
            try:
                points.append((position[0] + x, midline - int(value * amplitude)))
            except ValueError:
                points.append((position[0] + x, midline))

        big_ahh_cache[(filename, color, position, size)] = points
    points = big_ahh_cache[(filename, color, position, size)]
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 1)

def linear_interpolation(a, x, y):
    return x + (y - x) * (a)


class Button():
    def __init__(self, x, y, width, height, color, text, font_name, font_size, font_color, number=0):
        self.rect = pygame.Rect(x, y, width, height)
        self.original_col = color
        self.highlighted_col = tuple([max(0, self.original_col[i]-70) for i in range(3)])
        self.color = self.original_col
        self.text = text
        self.font = pygame.font.Font(font_name, font_size)
        self.pressed = False  # Track press state
        self.clicked = False  # Track full click cycle
        self.textcol = font_color
        self.number = number
        self.disabled = False

    def draw(self, surface):
        # Draw the button
        pygame.draw.rect(surface, self.color, self.rect, border_radius=8)

        # Render text and center it
        text_surf = self.font.render(self.text, True, self.textcol)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

        # Handle click detection
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        if not self.disabled:
            if self.rect.collidepoint(mouse_pos):
                self.color = self.highlighted_col
                if mouse_pressed and not self.pressed:
                    self.pressed = True  # Mouse went down
                elif not mouse_pressed and self.pressed:
                    self.pressed = False
                    return True  # Mouse went up — click complete
            else:
                self.color = self.original_col
                if not mouse_pressed:
                    self.pressed = False  # Reset if released off-button
        else:
            if self.rect.collidepoint(mouse_pos):
                screen.blit(nope_img, mouse_pos)
        return False

class Slider:
    def __init__(self, x, y, width, height, color, name, size=0.5, number=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.name = name
        self.size = max(0, min(1, size))  # Clamp between 0 and 1
        self.handle_radius = height // 2
        self.dragging = False
        self.font = pygame.font.SysFont(None, 24)
        self.number = number

    def draw(self, surface):
        # # Draw label
        # label = self.font.render(f"{self.name}: {math.ceil(self.size*100)}%", True, WHITE)
        # surface.blit(label, (self.x, self.y - 25))

        # Draw slider background
        pygame.draw.rect(surface, (180, 180, 180), (self.x, self.y, self.width, self.height), border_radius=5)

        # Draw slider fill
        fill_width = int(self.size * self.width)
        pygame.draw.rect(surface, self.color, (self.x, self.y, fill_width, self.height), border_radius=5)

        # Draw handle
        handle_x = self.x + fill_width
        handle_y = self.y + self.height // 2
        pygame.draw.circle(surface, (0, 0, 0), (handle_x, handle_y), self.handle_radius)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            handle_x = self.x + int(self.size * self.width)
            handle_y = self.y + self.height // 2
            if (mx - handle_x) ** 2 + (my - handle_y) ** 2 <= self.handle_radius ** 2:
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx, _ = event.pos
            relative_x = mx - self.x
            self.size = max(0, min(1, relative_x / self.width))  # Clamp

    def get_value(self):
        return self.size

master_pan = 0.5
master_volume = 0.75

TRIM_REGION = 25

pygame.mixer.set_num_channels(CHANNELS+2)
aux_channel = pygame.mixer.Channel(CHANNELS)
preview_channel = pygame.mixer.Channel(CHANNELS+1)
channels = []
for i in range(CHANNELS):
    channels.append(pygame.mixer.Channel(i))
def copy_audio(input_path, output_path):
    # Read the audio file
    data, sample_rate = sf.read(input_path)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to the new location
    sf.write(output_path, data, sample_rate)
class Audio(pygame.sprite.Sprite):
    def __init__(self, path, x, y):
        self.ready = False
        audio_clips.append(self)
        self.path = path
        self.name = "".join(path.split(".")[:-1])
        print(path, "t.se")
        self.original_path = "".join(path.split(".")[:-1])+"-OG0.wav"
        copy_audio(self.path, self.original_path)
        self.sound = pygame.mixer.Sound(path)
        self.playing = False
        self.fade_x = 0
        self.sound_length = self.sound.get_length()
        self.rect = pygame.Rect(x,y,self.sound.get_length()*LPS*(seq.tempo/240), TRACK_HEIGHT)
        self.grabbed = False
        self.track = 0
        self.x = x
        self.set_color()
        self.semitones = 0.00
        self.display_semitones = 0.00
        self.change_by = None
        self.stretching = False
        self.single_selected = True
        self.speed_rate = 1
        self.ready = True

        self.pmx = 0
        self.pmy = 0

        self.reapply()



    def set_color(self):
        self.color = colors[self.track % 8] if not muted[self.track] else (200, 200, 200)
        self.highlighted_color = tuple([min(255, self.color[i]+100) for i in range(3)])
        self.dark_col = tuple([max(0, self.color[i]-100) for i in range(3)])

    def quantize(self):
        if not quant: return
        if not self.grabbed:
            try:
                xx = bars[bisect.bisect(bars, self.x)]
                if xx-QUANT_MARGIN <= self.x <= xx+QUANT_MARGIN:self.x = xx
            except IndexError: pass
            try:
                xx = bars[bisect.bisect(bars, self.x)-1]
                if xx-QUANT_MARGIN <= self.x <= xx+QUANT_MARGIN:self.x = xx
            except IndexError: pass

    # def touching_edges(self):
    #     mp = pygame.mouse.get_pos()
    #     if self.rect.collidepoint(mp):
    #         wx = self.x+self.rect.width
    #         if (self.x <= mp[0] <= (self.x+TRIM_REGION)) or ((wx-TRIM_REGION) <= mp[0] <= wx):
    #             return True
    #     return False

    # def fade_check(self):
    #     mp = pygame.mouse.get_pos()
    #     if pygame.mouse.get_pressed()[0]:
    #         if self.rect.collidepoint(mp):
    #             wx = self.x+self.rect.width
    #             if (((wx-self.fade_x)-TRIM_REGION) <= mp[0] <= (wx)):
    #                 self.fade_x = wx - mp[0]
    #                 return True
    #     return False

    # def fade_check(self):
    #     if (not (stretch_mode or auto)) and my_plug=="None":
    #         mp = pygame.mouse.get_pos()
    #         if pygame.mouse.get_pressed()[0]:
    #             if self.rect.collidepoint(mp):
    #                 if self.rect.y <= mp[1] <= self.rect.y + (self.rect.height // 3):
    #                     if (self.x <= mp[0] <= (self.x+TRIM_REGION+self.fade_x)):
    #                         self.fade_x = mp[0] - self.x
    #                         return True
    #     return False


    def stretch(self, scale):
        self.speed_rate *= scale
        self.display_semitones += semi_formula(1/scale)
        self.reapply()
        # self.semitones += semi_formula(1/self.speed_rate)
        


    # def stretchTo(self, scale):
    #     """
    #     Stretch an audio file by a scale factor (affects speed and pitch).
        
    #     Parameters:
    #         input_path (str): Path to the input audio file.
    #         output_path (str): Path to save the stretched audio file.
    #         scale (float): Stretch factor (>1.0 = slower, <1.0 = faster).
    #     """
    #     audio = AudioSegment.from_file(self.path)
    #     samples = np.array(audio.get_array_of_samples())

    #     # For stereo, reshape and resample both channels
    #     if audio.channels == 2:
    #         samples = samples.reshape((-1, 2))
    #         left = resample(samples[:, 0], int(len(samples) / scale))
    #         right = resample(samples[:, 1], int(len(samples) / scale))
    #         new_samples = np.column_stack((left, right)).astype(samples.dtype)
    #     else:
    #         new_samples = resample(samples, int(len(samples) / scale)).astype(samples.dtype)

    #     new_audio = AudioSegment(
    #         new_samples.tobytes(),
    #         frame_rate=int(audio.frame_rate / scale),
    #         sample_width=audio.sample_width,
    #         channels=audio.channels
    #     )

    #     os.makedirs(os.path.dirname(self.path), exist_ok=True)
    #     new_audio.export(self.path, format=os.path.splitext(self.path)[1][1:])

    def stretchTo(self, scale):
        scale = 1/scale 
        audio = AudioSegment.from_file(self.path)
        samples = np.array(audio.get_array_of_samples())

        # Resample based on channel
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            left = resample(samples[:, 0], int(len(samples) * scale))
            right = resample(samples[:, 1], int(len(samples) * scale))
            new_samples = np.stack((left, right), axis=-1).astype(np.int16)
            new_samples = new_samples.flatten()
        else:
            new_samples = resample(samples, int(len(samples) * scale)).astype(np.int16)

        # Rebuild audio segment with ORIGINAL frame rate
        new_audio = AudioSegment(
            new_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )

        # Ensure output folder exists
        out_folder = os.path.dirname(self.path)
        if out_folder:
            os.makedirs(out_folder, exist_ok=True)

        # Export
        new_audio.export(self.path, format=self.path.split('.')[-1])
        
            
    def pitchTo(self, by):
        with AudioFile(self.path) as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate
        board = Pedalboard([PitchShift(semitones=by)]) 
        effected = board(audio, samplerate)
        with AudioFile(self.path, 'w', samplerate, effected.shape[0]) as f:
            f.write(effected)

    def pitch(self, by):
        self.semitones = (self.semitones+by)
        self.display_semitones += by
        self.reapply()

    def timeStretch(self, rate):
        self.speed_rate *= rate
        self.semitones += semi_formula(rate)
        self.reapply()

    def reverse(self):
        rate, data = wavfile.read(self.path)
        reversed_data = data[::-1]
        wavfile.write(self.path, rate, reversed_data)
        self.reapply()

    def reapply(self):
        reset_clips_playing()
        copy_audio(self.original_path, self.path)
        if studio_effects["EQ"]: process_wav_file(self.path, self.path, eq_values[self.track])
        if studio_effects["Reverb"]: process_reverb(self.path, self.path, reverb_values[self.track][:3], reverb_values[self.track][-2]*15, reverb_values[self.track][-1]*15)
        if studio_effects["Delay"]: process_delay(self.path, self.path, delay_values[self.track])
        if studio_effects["Compressor"]: apply_compression(self.path, self.path, compressor_values[self.track])
        if studio_effects["Gate"]: apply_gain_and_noise_gate(self.path, self.path, gate_values[self.track])
        if studio_effects["Chorus"]: apply_chorus(self.path, self.path, chorus_values[self.track])

        apply_volume(self.path, self.path, mixer_volumes[self.track]*1.5*master_volume)
        apply_pan(self.path, self.path, min(1, max(0, (mixer_pan[self.track]+master_pan) / 2 )))

        if self.semitones != 0:
            self.pitchTo(self.semitones)
        self.stretchTo(self.speed_rate)
        self.sound = pygame.mixer.Sound(self.path)
        self.sound_length = self.sound.get_length()

    def on_stretch(self):
        if stretch_mode:
            mp = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed()[0]:
                if self.rect.collidepoint(mp):
                    wx = self.x+self.rect.width
                    if (((wx-self.fade_x)-TRIM_REGION) <= mp[0] <= (wx)):
                        self.stretching = True
                
                if self.stretching:
                    self.change_by = (self.rect.width)/(max(mp[0]-self.x, 0.01))
                    line_x = self.x + (self.rect.width*(1/self.change_by))
                    pygame.draw.line(screen, GREEN, (line_x, MENU_OFFSET), (line_x, HEIGHT), 2)
                    return True
        if self.change_by != None:
            self.stretch(self.change_by)
            self.stretching = False
        self.change_by = None
        return False


    def update(self):
        global anything_grabbed, last_clicked
        try: self.track = ((self.rect.y-MENU_OFFSET) // TRACK_HEIGHT)
        except AttributeError: pass
        mp = pygame.mouse.get_pos()
        pointing = self.rect.collidepoint(mp)
        self.set_color()
        col = self.highlighted_color if pointing else self.color
        pygame.draw.rect(screen, col, self.rect)
        # pts = [(self.x + self.fade_x, self.rect.y), (self.x, self.rect.y), (self.x, self.rect.y+self.rect.height)]
        # pygame.draw.polygon(screen, self.dark_col, pts)
        touching_seq = pygame.sprite.collide_rect(self, seq)
        draw_waveform(self.path, screen, BLACK, self.rect.topleft, (self.rect.width, self.rect.height))
        # pygame.draw.circle(screen, WHITE, (pts[0][0], pts[0][1]+5), 10)

        if self.grabbed or last_clicked==self:
            pygame.draw.rect(screen, WHITE, self.rect, 4)

        w = (self.rect.width) // 20
        name = self.name
        if len(self.name) > w:
            name = f"{self.name[:w]}...{self.name[-3:]}"
        if self.rect.width <= 30:
            name = f"{self.name[0]}..."
        draw_text(screen, name, font10, self.rect.topleft, WHITE)

        self.rect.width = (self.sound_length*LPS) * (seq.tempo / 240)

        # p = (pan_slider.get_value() * 200) - 100
        # if p != self.panning:
        #     self.pan(max(-90, min(p, 90)))
        
        self.on_stretch()
        if (not (stretch_mode or auto)) and (my_plug=="None"):
            if pointing:
                if mp[0] >= X_OFF:
                    if pygame.mouse.get_pressed()[0]:
                        if not anything_grabbed:
                            self.grabbed = True
                            self.single_selected = True
                            last_clicked = self
                            anything_grabbed = True
                    if pygame.mouse.get_pressed()[2]:
                        audio_clips.remove(self)
                        last_clicked = None

        if (mp[1]-self.pmy == 0) and (not pygame.mouse.get_pressed()[0]):
            self.rect.y = max(((self.rect.y-MENU_OFFSET) // TRACK_HEIGHT * TRACK_HEIGHT) + MENU_OFFSET, MENU_OFFSET)

        self.x += dmx
        if self.grabbed:
            aOkay = True
            if not self.single_selected: aOkay = pygame.mouse.get_pressed()[0]
            if aOkay:
                if (main_point == [None, None]) or (not self.single_selected):
                    self.x += mp[0]-self.pmx
                    # self.rect.centerx += mp[0]-self.pmx
                    self.rect.y += mp[1]-self.pmy
                    if (not pygame.mouse.get_pressed()[0]) and self.single_selected:
                        self.grabbed = False
                        anything_grabbed = False
                else:
                    self.grabbed = False
                    anything_grabbed = False
        
        if not self.playing:
            if seq.on:
                if touching_seq:
                    if not muted[self.track]:
                        # Calculate offset in seconds from start of clip to sequence line
                        offset_px = seq.rect.x - self.x
                        offset_sec = offset_px / (LPS * (seq.tempo / 240))
                        if offset_sec > 0.01:  # If not at the very start
                            
                            # Create a temp wav file starting from offset
                            audio = AudioSegment.from_wav(self.path)
                            trimmed = audio[offset_sec * 1000:]
                            temp_name = f"temps/temp_{uuid.uuid4().hex}.wav"
                            trimmed.export(temp_name, format="wav")
                            self.temp_playing_path = temp_name
                            temp_sound = pygame.mixer.Sound(temp_name)
                            channels[self.track].play(temp_sound, fade_ms=int((self.fade_x/LPS)*1000))
                            self.playing = True
                            os.remove(temp_name)
                        else:
                            channels[self.track].play(self.sound, fade_ms=int((self.fade_x/LPS)*1000))
                            self.playing = True
        else:
            if not touching_seq:
                self.playing = (self.sound.get_num_channels() > 0)
                # Clean up temp file if it exists
                if hasattr(self, 'temp_playing_path'):
                    try:
                        os.remove(self.temp_playing_path)
                    except Exception:
                        pass
                    del self.temp_playing_path
        self.pmx = mp[0]
        self.pmy = mp[1]
        self.rect.x = self.x
        self.quantize()
            
last_clicked = None

studio_effects = {
    "EQ": False,
    "Reverb": False,
    "Delay": False,
    "Compressor": False,
    "Gate": False,
    "Chorus": False,

}

muted = [False]*CHANNELS
soloed = [False]*CHANNELS

stretch_mode = False

RS["tt"] = 0

class Seque(pygame.sprite.Sprite):
    def __init__(self):
        self.on = False
        self.rect = pygame.Rect(X_OFF, MENU_OFFSET, 2, TRACK_HEIGHT*CHANNELS)
        self.pushf = 0
        self.grabbed = False
        self.last_x = float("inf")
        self.tempo = 120

    def quantize(self):
        if quant and self.grabbed:
            try: 
                xx = bars[bisect.bisect(bars, seq.rect.x)]
            except IndexError: pass
            else:
                if xx-QUANT_MARGIN <= seq.rect.x <= xx+QUANT_MARGIN:
                    seq.rect.x = xx
            try: 
                xx = bars[bisect.bisect(bars, seq.rect.x)-1]
            except IndexError: pass
            else:
                if xx-QUANT_MARGIN <= seq.rect.x <= xx+QUANT_MARGIN:
                    seq.rect.x = xx

    def update(self):
        pygame.draw.rect(screen, WHITE, self.rect)
        self.rect.x += dmx
        if self.on:
            self.pushf += LPS * dt * (self.tempo / 240)
            if self.pushf >= 1:
                self.rect.x += int(self.pushf)
                self.pushf -= int(self.pushf)
        if dmx == 0:
            if metro:
                if not ((MENU_OFFSET-30 <= my <= MENU_OFFSET) and pygame.mouse.get_pressed()[0]):
                    if int((self.rect.x - acc - X_OFF) % (LPS)) < int((self.last_x - acc - X_OFF) % (LPS)):
                        aux_channel.play(pygame.mixer.Sound("sfx/beat.wav"))
                    elif int((self.rect.x - acc - X_OFF) % (LPS / maxa)) < int((self.last_x - acc - X_OFF) % (LPS / maxa)):
                        aux_channel.play(pygame.mixer.Sound("sfx/bar.wav"))
        self.quantize()
        self.last_x = self.rect.x

    def run(self):
        self.on = True

    def stop(self):
        reset_clips_playing()
        self.on = False
        pygame.mixer.stop()
        for audio in audio_clips:
            self.playing = False

automations = []
for i in range(CHANNELS):
    automations.append({"volume": [], "pan": []})

class AutoPoint(pygame.sprite.Sprite):
    def __init__(self, x, y, track, effect):
        self.groups = ()
        pygame.sprite.Sprite.__init__(self, self.groups)
        global automations
        automations[track][effect].append(self)
        self.pos = [x, y]
        self.track = track
        self.type = effect
        self.grabbed = False
        self.radius = 8
        self.get_rect()

    def get_rect(self):
        self.rect = pygame.Rect(self.pos[0]-(self.radius // 2), self.pos[1]-(self.radius // 2), self.radius, self.radius)

    def clicked(self):
        return self.rect.collidepoint(list(pygame.mouse.get_pos())) and pygame.mouse.get_pressed()[0]
    
    def update(self):
        mouse_pos = list(pygame.mouse.get_pos())
        pressed = pygame.mouse.get_pressed()[0]
        pointing = self.rect.collidepoint(mouse_pos)
        self.pos[0] += dmx

        if self.clicked():
                self.grabbed = True
        elif self.rect.collidepoint(list(pygame.mouse.get_pos())) and pygame.mouse.get_pressed()[2]:
            automations[self.track][self.type].remove(self)
        else:
            if not pressed:
                self.grabbed = False

        if self.grabbed:
            self.pos = mouse_pos.copy()
        self.pos[1] = clamp(self.pos[1], MENU_OFFSET+self.track*TRACK_HEIGHT, MENU_OFFSET+(self.track+1)*TRACK_HEIGHT)

        self.get_rect()

        if auto:
            pygame.draw.circle(screen, WHITE, self.pos, 4)
            pygame.draw.circle(screen, BLACK, self.pos, 4, 1)

paths_to_save = set()

def load_audio():
    filepath = filedialog.askopenfilename(
            title="Select Code File",
            filetypes=(
                ("Wave files", "*.wav"),
            )
        )
    if filepath:
        button = Button(5, MENU_OFFSET+10+55*len(paths_to_save)+15, X_OFF-10, 35, WHITE, "  ", font_name, 15, BLACK, )
        paths_to_save.add((filepath, button))
        spawn_clip(filepath)

def spawn_clip(filepath):
    global RANDOM_CONSTANT
    destination_path = f'sounds/{RANDOM_CONSTANT}d-{filepath.split('/')[-1]}'
    shutil.copy(filepath, destination_path)
    a = Audio(destination_path, seq.rect.x, (selected_track*TRACK_HEIGHT)+MENU_OFFSET)
    a.name = "".join(filepath.split('/')[-1].split(".")[:-1])
    RANDOM_CONSTANT+=1

font_name = None
show_paths = False
metro_slider = Slider(10, 20, 110, 15, BUTTON_COL, "Metronome Volume")
metro_button = Button(50, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)

tempo_button = Button(100, 40, 80, 40, BUTTON_COL, "-- bpm", font_name, 25, WHITE)
tempo_plus = Button(180, 40, 20, 20, BUTTON_COL, "+", font_name, 25, WHITE)
tempo_minus = Button(180, 60, 20, 20, BUTTON_COL, "-", font_name, 25, WHITE)

run_button = Button(210, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
record_button = Button(250, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
reset_button = Button(290, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)

up_button = Button(340, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
quant_button = Button(385, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
split_button = Button(430, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
download_button = Button(475, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
align_tempo = Button(520, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
# auto_button = Button(565, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
mixer_button = Button(565, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
time_button = Button(610, 40, 100, 40, BUTTON_COL, "  ", font_name, 25, WHITE)

consolidate_button = Button(715, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
plugin_button = Button(760, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
bpb_button = Button(805, 40, 80, 40, BUTTON_COL, "  ", font_name, 25, WHITE)

eq_button = Button(210, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
reverb_button = Button(255, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
delay_button = Button(300, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
compressor_button = Button(345, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
gate_button = Button(390, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)
chorus_button = Button(435, 40, 40, 40, BUTTON_COL, "  ", font_name, 25, WHITE)

apply_button = Button(WIDTH-250, MENU_OFFSET+50, 200, 50, WHITE, "Apply", font_name, 30, BLACK)

mixer_apply_button = Button(WIDTH-210, 10, 200, 35, WHITE, "Apply", font_name, 25, BLACK)

leave_button = Button(WIDTH-420, 10, 200, 35, WHITE, "Exit Mixer", font_name, 25, BLACK)

on_button = Button(X_OFF+20, MENU_OFFSET+20, 50, 25, WHITE, "On", font_name, 30, BLACK)

# zi_button = Button(840, 40, 80, 40, BUTTON_COL, "zoom in", font_name, 25, WHITE)
# zo_button = Button(930, 40, 80, 40, BUTTON_COL, "zoom out", font_name, 25, WHITE)

nope_img = Image("images/nope.svg")
bpm_taps = []

RANDOM_CONSTANT = 0

def split():
    global RANDOM_CONSTANT
    copyci = audio_clips.copy()
    for audio in copyci:
        if audio.track == selected_track:
            if audio.x+5 <= seq.rect.x <= audio.x+audio.rect.width-5:
                audio_clips.remove(audio)
                seconds = (seq.rect.x - audio.x) / (LPS * (seq.tempo / 240))
                sound = AudioSegment.from_file(audio.path)

                name = f"{audio.name}-s{RANDOM_CONSTANT}.wav"
                clip = sound[seconds*1000:]
                clip.export(name, format="wav")
                Audio(name, seq.rect.x, (selected_track*TRACK_HEIGHT)+MENU_OFFSET)
                RANDOM_CONSTANT+=1

                clip = sound[:seconds*1000]
                name = f"{audio.name}-s{RANDOM_CONSTANT}.wav"
                clip.export(name, format="wav")
                Audio(name, audio.x, (selected_track*TRACK_HEIGHT)+MENU_OFFSET)
                RANDOM_CONSTANT+=1
                
quant = True

selected_track = 0
mouse_offset = 0
dmx = 0
acc = 0
bar_info_height = 30
# Main loop
running = True
seq = Seque()

#maximum bars per section 
maxa = 4
not_pressed_before = False
anything_grabbed = False
QUANT_MARGIN = 20 * (LPS//200)

sliders = []
mute_buttons = []
solo_buttons = []
for i in range(CHANNELS):
    sliders.append(Slider(10, MENU_OFFSET+(i*TRACK_HEIGHT)+20, X_OFF-70, 15, (150,150,150), f"Track {i+1} Vol", number=i))
    mute_buttons.append(Button(95, MENU_OFFSET+(i*TRACK_HEIGHT)+17, 20, 20, BUTTON_COL, "M", font_name, 16, WHITE, number=i))
    solo_buttons.append(Button(120, MENU_OFFSET+(i*TRACK_HEIGHT)+17, 20, 20, BUTTON_COL, "S", font_name, 16, WHITE, number=i))

pitch_plus = Button(10, MENU_OFFSET + 20, 40, 40, BUTTON_COL, "+1", font_name, 25, WHITE)
pitch_minus = Button(100, MENU_OFFSET + 20, 40, 40, BUTTON_COL, "-1", font_name, 25, WHITE)
pitch_fplus = Button(10, MENU_OFFSET + 70, 40, 40, BUTTON_COL, "+0.25", font_name, 25, WHITE)
pitch_fminus = Button(100, MENU_OFFSET + 70, 40, 40, BUTTON_COL, "-0.25", font_name, 25, WHITE)

plugins = False

speed_plus = Button(10, MENU_OFFSET + 140, 40, 40, BUTTON_COL, "+5 %", font_name, 25, WHITE)
speed_minus = Button(100, MENU_OFFSET + 140, 40, 40, BUTTON_COL, "-5 %", font_name, 25, WHITE)
speed_fplus = Button(10, MENU_OFFSET + 190, 40, 40, BUTTON_COL, "+1 %", font_name, 25, WHITE)
speed_fminus = Button(100, MENU_OFFSET + 190, 40, 40, BUTTON_COL, "-1 %", font_name, 25, WHITE)

# pan_slider = Slider(10, MENU_OFFSET+120, X_OFF-25, 20, WHITE, "Pan")

recordX = None
recording = False

colors = [
    (255, 0, 0),       # Vibrant Red
    (255, 102, 0),     # Vibrant Orange
    (255, 255, 0),     # Vibrant Yellow
    (0, 255, 0),       # Vibrant Green
    (0, 255, 255),     # Vibrant Cyan
    (0, 102, 255),     # Vibrant Blue
    (255, 0, 255),      # Vibrant Magenta
    (191, 0, 255),     # Vibrant Purple

]

def by_x(x):
    return x.pos[0]

def consolidate():
    global selected_clips, many_sel, anything_grabbed, last_clicked, RANDOM_CONSTANT
    if many_sel:
        # Get all selected clips
        selected_clips = [audio for audio in audio_clips if audio.grabbed and not audio.single_selected]
        if len(selected_clips) > 1:
            # Find the highest track and leftmost/rightmost positions
            highest_track = max(clip.track for clip in selected_clips)
            leftmost_x = min(clip.rect.x for clip in selected_clips)
            rightmost_x = max(clip.rect.x + clip.rect.width for clip in selected_clips)
            
            # Create a temporary directory for processing
            temp_dir = "temp_consolidate"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create a silent segment of the total length
            total_length = (rightmost_x - leftmost_x) / (LPS * (seq.tempo / 240)) * 1000  # Convert to milliseconds
            combined = AudioSegment.silent(duration=total_length)
            
            # Overlay each clip at its correct position
            for clip in selected_clips:
                # Calculate the offset in milliseconds
                offset = (clip.rect.x - leftmost_x) / (LPS * (seq.tempo / 240)) * 1000
                
                # Load the audio file
                sound = AudioSegment.from_wav(clip.path)
                
                # Overlay the sound at the correct position
                combined = combined.overlay(sound, position=int(offset))
            
            # Export the combined file
            output_path = f"sounds/consolidated_{RANDOM_CONSTANT}.wav"
            combined.export(output_path, format="wav")
            
            # Create new audio clip
            Audio(output_path, leftmost_x, (highest_track * TRACK_HEIGHT) + MENU_OFFSET)
            RANDOM_CONSTANT += 1
            
            # Remove original clips
            for clip in selected_clips:
                audio_clips.remove(clip)
            
            # Clean up temp directory
            os.rmdir(temp_dir)
            
            # Reset selection state
            many_sel = False
            anything_grabbed = False
            last_clicked = None
main_point = [None, None]
many_sel = False
highlight_rect = pygame.Rect(
            0, 0, 0, 0
            )
copied_audio = {"full": [], "min x": 0, "min track": 0}
def record():
    global recording, recordX, audio, q, seq, RANDOM_CONSTANT, audio, audio_frames
    if not recording:
                    seq.run()
                    recording = True
                    recordX = seq.rect.x
                    audio_frames = []
                    with q.mutex:
                        q.queue.clear()  # clear old audio
    else:
        seq.stop()
        recording = False
        while not q.empty():
            audio_frames.append(q.get())
        if audio_frames:
            audio = np.concatenate(audio_frames, axis=0)
            write(f"sounds/recording{RANDOM_CONSTANT}.wav", samplerate, audio)
            Audio(f"sounds/recording{RANDOM_CONSTANT}.wav", recordX, selected_track*TRACK_HEIGHT + MENU_OFFSET)
            RANDOM_CONSTANT+=1

start_stream()
my_plug = "None"
metro = False

auto = False

stretch_mouse_img = Image("images/stretch_mouse.svg")
pause_img = Image("images/pause.svg")
play_img = Image("images/play.svg")
record_img = Image("images/record.svg")
end_record_img = Image("images/end_record.svg")
metro_on_img = Image("images/metro_on.svg")
metro_off_img = Image("images/metro_off.svg")
reset_img = Image("images/reset.svg")

dnld_img = Image("images/download.svg")
free_img = Image("images/free.svg")
quant_img = Image("images/quant.svg")
upload_img = Image("images/upload.svg")
split_img = Image("images/split.svg")
align_tempo_img = Image("images/align_tempo.svg")
automation_img = Image("images/auto.svg")
no_automation_img = Image("images/no_auto.svg")
mixer_img = Image("images/mixer.svg")
consolidate_img = Image("images/consolidate.svg")
plugin_img = Image("images/plug.svg")
eq_img = Image("images/eq.svg")
reverb_img = Image("images/reverb.svg")
delay_img = Image("images/delay.svg")
compressor_img = Image("images/compression.svg")
gate_img = Image("images/gate.svg")
chorus_img = Image("images/chorus.svg")

bope_img = Image("images/bope.svg")

def reset_clips_playing():
    for audio in audio_clips:
        audio.playing = False

old_bpm = seq.tempo

def update_audio():
    global audio_clips
    for audio in audio_clips:
        if audio.ready:
            audio.update()

RS["oi"] = 0

class Knob:
    def __init__(self, x, y, radius, color, value=0.5, sense=1):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.sense = sense
        self.value = value  # 0 to 1
        self.angle = -135 + (value * 270)  # Convert value to angle (-135 to 135)
        self.grabbed = False
        self.last_y = None

    def draw(self, screen):
        # Draw knob base
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
        pygame.draw.circle(screen, (0, 0, 0), (self.x, self.y), self.radius, 2)
        
        # Draw indicator line
        end_x = self.x + (self.radius * 0.8 * math.cos(math.radians(self.angle)))
        end_y = self.y + (self.radius * 0.8 * math.sin(math.radians(self.angle)))
        pygame.draw.line(screen, (255, 255, 255), (self.x, self.y), (end_x, end_y), 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = pygame.mouse.get_pos()
                distance = math.sqrt((mouse_pos[0] - self.x)**2 + (mouse_pos[1] - self.y)**2)
                if distance <= self.radius:
                    self.grabbed = True
                    self.last_y = mouse_pos[1]
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click release
                self.grabbed = False
                self.last_y = None
        
        elif event.type == pygame.MOUSEMOTION and self.grabbed:
            mouse_pos = pygame.mouse.get_pos()
            if self.last_y is not None:
                # Calculate change in y position
                dy = self.last_y - mouse_pos[1]
                # Convert to angle change (adjust sensitivity as needed)
                angle_change = dy * self.sense
                
                self.angle += angle_change
                self.angle = max(-135, min(135, self.angle))  # Clamp angle
                self.value = (self.angle + 135) / 270  # Convert angle back to value
                self.last_y = mouse_pos[1]

mixer = False

# Add this after the other button initializations
# Initialize EQ knobs
knob_radius = 30
knob_spacing = 80
paths_y = 0
eq_knobs = []
eq_values = []
for i in range(CHANNELS): 
    eq_values.append([0.5] * 7)
    eq_knobs.append([])
for e in range(CHANNELS):
    start_x = X_OFF + 50
    start_y = MENU_OFFSET + 100
    for i in range(7):
        x = start_x + (i * knob_spacing)
        eq_knobs[e].append(Knob(x, start_y, knob_radius, (200, 50, 50), 0.5))

reverb_knobs = []
reverb_values = []
for i in range(CHANNELS): 
    reverb_values.append([0,0,0,0.2,0])
    reverb_knobs.append([])
for e in range(CHANNELS):
    start_x = X_OFF + 50
    start_y = MENU_OFFSET + 100
    for i in range(5):
        x = start_x + (i * (knob_spacing * 7/5))
        reverb_knobs[e].append(Knob(x, start_y, knob_radius, "#9e54ff", reverb_values[e][i]))

mixer_volumes = [0.5]*CHANNELS
left_sidebar = pygame.Rect(0, MENU_OFFSET-30, X_OFF, HEIGHT)
mx, my = 0, 0
delay_knobs = []
delay_values = []
for i in range(CHANNELS): 
    delay_values.append([0] * 3)
    delay_knobs.append([])
for e in range(CHANNELS):
    start_x = X_OFF + 50
    start_y = MENU_OFFSET + 100
    for i in range(3):
        x = start_x + (i * (knob_spacing * 7/3))
        delay_knobs[e].append(Knob(x, start_y, knob_radius, "#67fbff", 0))

insert_mode = True

compressor_knobs = []
compressor_values = []
for i in range(CHANNELS): 
    compressor_values.append([1, 0.5, 0, 0])
    compressor_knobs.append([])
for e in range(CHANNELS):
    start_x = X_OFF + 50
    start_y = MENU_OFFSET + 100
    for i in range(4):
        x = start_x + (i * (knob_spacing * 7/4))
        compressor_knobs[e].append(Knob(x, start_y, knob_radius, "#726E00", compressor_values[e][i]))

gate_knobs = []
gate_values = []
for i in range(CHANNELS): 
    gate_values.append([0.5, 0])
    gate_knobs.append([])
for e in range(CHANNELS):
    start_x = X_OFF + 50
    start_y = MENU_OFFSET + 100
    for i in range(2):
        x = start_x + (i * (knob_spacing * 7/2))
        gate_knobs[e].append(Knob(x, start_y, knob_radius, "#02c37d", gate_values[e][i]))

chorus_knobs = []
chorus_values = []
for i in range(CHANNELS): 
    chorus_values.append([0.3,0.8,10,0.3,0])
    chorus_knobs.append([])
for e in range(CHANNELS):
    start_x = X_OFF + 50
    start_y = MENU_OFFSET + 100
    for i in range(5):
        x = start_x + (i * (knob_spacing * 7/5))
        chorus_knobs[e].append(Knob(x, start_y, knob_radius, "#67FF00", chorus_values[e][i]))

channel_width = 75
pan_knobs = []
mixer_pan = []
plugin_mixer = []
for i in range(CHANNELS):
    pan_knobs.append(Knob(315 + (i*channel_width), X_OFF*0.5+90, 22, (60, 60, 60), 0.5, sense=5))
    mixer_pan.append(0.5)
    plugin_mixer.append(Button(299 + (i*channel_width), HEIGHT-97, 32, 32, colors[i%10], "+", font_name, 25, WHITE))
master_pan_knob = Knob(177, X_OFF*0.5+80, 28, (60, 60, 60), 0.5, sense=5)

while running:
    ticks = clock.tick(FPS)
    dt = ticks / 1000
    keys = pygame.key.get_pressed()


    if not mixer:
        screen.fill((120,120,120))
        QUANT_MARGIN = 20 * (LPS/200)
        # draw_text(screen, f"{clock.get_fps()}", font, (0, 0))

        # draw selected track shaded
        pygame.draw.rect(screen, (90,90,90), (X_OFF+acc, MENU_OFFSET + selected_track*TRACK_HEIGHT, MAXWIDTH, TRACK_HEIGHT))

        # draw bars / beats
        pygame.draw.rect(screen, (150, 150, 150), pygame.Rect(X_OFF+acc, MENU_OFFSET - bar_info_height, MAXWIDTH, bar_info_height))
        bars = []
        bar = 1
        x = X_OFF
        while x <= MAXWIDTH:
            draw_text(screen, f"{bar}", font, (x+acc, MENU_OFFSET-bar_info_height+5), WHITE)
            for i in range(maxa):
                xx = x+acc+(LPS/maxa * i)
                bars.append(xx)
                pygame.draw.line(screen, (150,150,150), (xx, MENU_OFFSET-bar_info_height+5), (xx, MENU_OFFSET+TRACK_HEIGHT*CHANNELS))
            x += LPS
            bar+=1
        bars.sort()

        pygame.draw.line(screen, (150,150,150), (x+acc, MENU_OFFSET-bar_info_height+5), (x+acc, MENU_OFFSET+TRACK_HEIGHT*CHANNELS))

        update_audio()
        
        # draw automation stuff
        for i in range(CHANNELS):
            pygame.draw.line(screen, BLACK, (X_OFF+acc, MENU_OFFSET + i*TRACK_HEIGHT), (WIDTH, MENU_OFFSET + i*TRACK_HEIGHT))
            if auto:
                s = pygame.Surface((WIDTH,TRACK_HEIGHT), pygame.SRCALPHA)
                s.fill((0,0,0,128))
                screen.blit(s, (X_OFF,MENU_OFFSET+i*TRACK_HEIGHT))
            for effect in automations[i]:
                # if effect == "volume":
                points = automations[i][effect]
                if len(points) > 0:
                    current_x = seq.rect.x
                    before_point = None
                    after_point = None
                    for point in points:
                        if point.pos[0] <= current_x:
                            before_point = point
                        else:
                            after_point = point
                            break
                    if before_point and after_point:
                        x1, y1 = before_point.pos
                        x2, y2 = after_point.pos
                        vol1 = 1 - (y1 - (MENU_OFFSET + i*TRACK_HEIGHT)) / TRACK_HEIGHT
                        vol2 = 1 - (y2 - (MENU_OFFSET + i*TRACK_HEIGHT)) / TRACK_HEIGHT
                        t = (current_x - x1) / (x2 - x1)
                        vol = vol1 + (vol2 - vol1) * t
                    elif before_point:
                        y = before_point.pos[1]
                        vol = 1 - (y - (MENU_OFFSET + i*TRACK_HEIGHT)) / TRACK_HEIGHT
                    elif after_point:
                        y = after_point.pos[1]
                        vol = 1 - (y - (MENU_OFFSET + i*TRACK_HEIGHT)) / TRACK_HEIGHT
                    else:
                        vol = 1.0 
                    # Clamp volume between 0 and 1
                    if effect == "volume":
                        vol = clamp(vol, 0, 1)
                        channels[i].set_volume(vol)
                    elif effect == "pan":
                        pan = clamp(vol, -1, 1)
                        set_pan(channels[i], pan)
                automations[i][effect].sort(key=by_x)
                
                if chosen_effect == effect:
                    if len(automations[i][effect]) != 0:
                        if auto:
                            pygame.draw.line(screen, WHITE, automations[i][effect][-1].pos, (WIDTH,automations[i][effect][-1].pos[1]), 1)
                            pygame.draw.line(screen, WHITE, automations[i][effect][0].pos, (0,automations[i][effect][0].pos[1]), 1)
                    for p, point in enumerate(automations[i][effect]):
                        if p != len(automations[i][effect])-1:
                            if auto:
                                pygame.draw.line(screen, WHITE, point.pos, automations[i][effect][p+1].pos, 2)
                        point.update()
                        

                    

        seq.grabbed = False
        # change selected track
        mx, my = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            if not_pressed_before:
                not_pressed_before = False
                good = True
                for audio in audio_clips:
                    if audio.rect.collidepoint([mx, my]):
                        good = False
                if good: main_point = [mx, my]
                else: main_point = [None, None]
            if MENU_OFFSET-30 <= my <= MENU_OFFSET:
                fine = True
                for audio in audio_clips:
                    if audio.rect.collidepoint([mx,my]):
                        fine = False
                if fine: 
                    pygame.mixer.stop()
                    seq.grabbed = True
                    if my_plug == "None":
                        seq.rect.x = mx
                        reset_clips_playing()
            elif my > MENU_OFFSET:
                fine = True
                for audio in audio_clips:
                    if audio.rect.collidepoint([mx,my]):
                        fine = False
                if fine:
                    if mx >= X_OFF:
                        if my_plug == "None":
                            selected_track = (my-MENU_OFFSET) // TRACK_HEIGHT
                        last_clicked = None
                    
        else:
            if not not_pressed_before:
                if highlight_rect.width > 20 and highlight_rect.height > 20:
                    for audio in audio_clips:
                        if pygame.Rect.colliderect(highlight_rect, audio.rect):
                            anything_grabbed = True
                            audio.grabbed = True
                            audio.single_selected = False
                            many_sel = True
            not_pressed_before = True
            main_point = [None, None]

        seq.update()
        if (not auto) and ((my_plug=="None")):
            if main_point != [None, None]:
                highlight_rect = pygame.Rect(
                    min(main_point[0], mx),
                    max(min(main_point[1], my), MENU_OFFSET),
                    max(main_point[0], mx)-min(main_point[0], mx),
                    max(main_point[1], my)-max(min(main_point[1], my), MENU_OFFSET),
                    )
                if highlight_rect.width > 20 and highlight_rect.height > 20:
                    if not many_sel:
                        pygame.draw.rect(screen, BLACK, highlight_rect, 8)
        if recording: pygame.draw.rect(screen, colors[selected_track % 8], (recordX, TRACK_HEIGHT*selected_track + MENU_OFFSET, seq.rect.x-recordX, TRACK_HEIGHT))
        
        if many_sel:
            pygame.draw.rect(screen, (140, 140, 140), pygame.Rect(mx-5, my-5, 275, 30))
            draw_text(screen, f"Press [Q] to exit from multi-select", font, (mx, my), WHITE)
        
        left_sidebar = pygame.Rect(0, MENU_OFFSET-30, X_OFF, HEIGHT)
        pygame.draw.rect(screen, (50, 50, 50), left_sidebar)

        if not plugins:
            if up_button.draw(screen):
                load_audio()
            if run_button.draw(screen):
                if seq.on:
                    seq.stop()
                else:
                    seq.run()
            if record_button.draw(screen):
                record()
            if reset_button.draw(screen):
                pygame.mixer.stop()
                seq.rect.x = acc + X_OFF - 10
                reset_clips_playing()

            metro_slider.draw(screen)
            if metro_button.draw(screen):
                metro = not metro
            if quant_button.draw(screen):
                quant = not quant
            if split_button.draw(screen):
                    split()
            split_button.disabled = (stretch_mode or auto)
            if download_button.draw(screen):
                renderer.render(audio_clips.copy(), LPS, -acc, X_OFF, automations, seq)
                renderer.dnld()
            
            if tempo_button.draw(screen): 
                now = time.time()
                bpm_taps.append(now)
                if len(bpm_taps) > 4:
                    intervals = [bpm_taps[i+1] - bpm_taps[i] for i in range(len(bpm_taps)-1)]
                    bpm = 60 / (sum(intervals) / len(intervals))
                    seq.tempo = bpm
                    if seq.tempo <= 40:
                        seq.tempo = 120
            tempo_button.text = f"{round(seq.tempo)} bpm"

            if align_tempo.draw(screen):
                for audio in audio_clips:
                    audio.timeStretch(1/(old_bpm/seq.tempo))
                old_bpm = seq.tempo

            tempo_plus.draw(screen)
            if pygame.mouse.get_pressed()[0] and tempo_plus.rect.collidepoint([mx, my]):
                RS["tt"] += RS["ttep"]
                if RS["tt"] > 3:
                    seq.tempo = min(seq.tempo+1, 4096); RS["tt"] = 0
                    RS["ttep"] *= 1.1
            else:
                RS["ttep"] = 1.01
            tempo_minus.draw(screen)
            if pygame.mouse.get_pressed()[0] and tempo_minus.rect.collidepoint([mx, my]):
                RS["tt"] += RS["ttem"]
                if RS["tt"] > 3:
                    seq.tempo = max(seq.tempo-1, 40); RS["tt"] = 0
                    RS["ttem"] *= 1.1
            else:
                RS["ttem"] = 1.01

            # if auto_button.draw(screen):
            #     auto = not auto
            #     pygame.mouse.set_visible(True)
            #     stretch_mode = False
            if mixer_button.draw(screen):
                mixer = True

            if consolidate_button.draw(screen):
                consolidate()

            if bpb_button.draw(screen):
                maxa -= 1
                if maxa == 0:
                    maxa = 8
            bpb_button.text = f"{maxa} bar{"s" if maxa > 1 else ""}"

            if time_button.draw(screen):
                pass
                # Calculate time passed based on sequence position
            time_passed = (seq.rect.x - (X_OFF + acc)) / (LPS * (seq.tempo / 240))
            time_button.text = (f"{time_format(int(time_passed//60))}:{time_format(int(time_passed % 60))}.{int((time_passed % 1) * 10)}s") if time_passed >= 0 else "00:00.0s"

            screen.blit(pause_img if seq.on else play_img, (210, 40))
            screen.blit(end_record_img if recording else record_img, (250, 40))
            screen.blit(reset_img, (290, 40))

            screen.blit(metro_on_img if metro else metro_off_img, (50, 40))

            screen.blit(upload_img, (340, 40))
            screen.blit(quant_img if quant else free_img, (385, 40))
            screen.blit(split_img, (430, 40))
            screen.blit(dnld_img, (475, 40))
            screen.blit(align_tempo_img, (520, 40))
            # screen.blit(no_automation_img if auto else automation_img, (565, 40))
            screen.blit(mixer_img, (565, 40))
            screen.blit(consolidate_img, (715, 40))
        else:
            if eq_button.draw(screen):
                if my_plug == "None": my_plug = "EQ"
                else: my_plug = "None"
            if reverb_button.draw(screen):
                if my_plug == "None": my_plug = "Reverb"
                else: my_plug = "None"
            if delay_button.draw(screen):
                if my_plug == "None": my_plug = "Delay"
                else: my_plug = "None" 
            if compressor_button.draw(screen):
                if my_plug == "None": my_plug = "Compressor"
                else: my_plug = "None" 
            if gate_button.draw(screen):
                if my_plug == "None": my_plug = "Gate"
                else: my_plug = "None" 
            if chorus_button.draw(screen):
                if my_plug == "None": my_plug = "Chorus"
                else: my_plug = "None"

            screen.blit(eq_img if my_plug == "None" else bope_img, (210, 40))
            screen.blit(reverb_img if my_plug == "None" else bope_img, (255, 40))
            screen.blit(delay_img if my_plug == "None" else bope_img, (300, 40))
            screen.blit(compressor_img if my_plug == "None" else bope_img, (345, 40))
            screen.blit(gate_img if my_plug == "None" else bope_img, (390, 40))
            screen.blit(chorus_img if my_plug == "None" else bope_img, (435, 40))

            if my_plug == "EQ":
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                s.fill((200, 50, 50, 128))
                screen.blit(s, (X_OFF, MENU_OFFSET-30))
                
                draw_text(screen, "Seven-Band Metric EQ", font35, (X_OFF + 90, MENU_OFFSET + 20), WHITE)
                freqs = ["100Hz", "200Hz", "400Hz", "800Hz", "1.6kHz", "3.2kHz", "6.4kHz"]
                for i, freq in enumerate(freqs):
                    x = X_OFF + 45 + (i * 80)
                    draw_text(screen, freq, font, (x - 20, MENU_OFFSET + 140), WHITE)
                for i, knob in enumerate(eq_knobs[selected_track]):
                    knob.draw(screen)
                    eq_values[selected_track][eq_knobs[selected_track].index(knob)] = 1 - knob.value
                    x = X_OFF + 45 + (i * 80)
                    draw_text(screen, f"{round((1 - knob.value)*100, 1)}%", font, (x - 20, MENU_OFFSET + 170), WHITE)

                on_button.text = "On" if studio_effects[my_plug] else "Off"
                if on_button.draw(screen):
                    studio_effects[my_plug] = not studio_effects[my_plug]

                if apply_button.draw(screen):
                    for audio in audio_clips:
                        if audio.track == selected_track:
                            audio.reapply()
                # if eq_napply_button.draw(screen):
                #     for audio in audio_clips:
                #         if audio.track == selected_track:
                #             process_wav_file(audio.original_path, audio.path, eq_values[selected_track], normalize=True)
                #             audio.reapply()

                # Add EQ visualizer
                draw_eq_visualizer(screen, X_OFF + 50, MENU_OFFSET + 250, WIDTH - X_OFF - 100, 150, eq_values[selected_track])

            if my_plug == "Reverb":
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                s.fill((78, 0, 181, 128))
                screen.blit(s, (X_OFF, MENU_OFFSET-30))
                names = ["Size", "Mix", "Tone", f"Tail ({round(reverb_values[selected_track][-2]*15, 1)} sec)", f"Delay ({round(reverb_values[selected_track][-1]*15, 2)} sec)"]
                for i, name in enumerate(names):
                    x = X_OFF + 45 + (i * 80 * 7/5)
                    draw_text(screen, name, font, (x - 20, MENU_OFFSET + 140), WHITE)
                for i, knob in enumerate(reverb_knobs[selected_track]):
                    knob.draw(screen)
                    reverb_values[selected_track][reverb_knobs[selected_track].index(knob)] = knob.value
                    x = X_OFF + 45 + (i * 80 * 7/5)
                    draw_text(screen, f"{round((knob.value)*100, 1)}%", font35, (x - 20, MENU_OFFSET + 170), WHITE)

                draw_reverb_visualizer(screen, X_OFF + 50, MENU_OFFSET + 250, WIDTH - X_OFF - 100, 150, reverb_values[selected_track])
                
            

                on_button.text = "On" if studio_effects[my_plug] else "Off"
                if on_button.draw(screen):
                    studio_effects[my_plug] = not studio_effects[my_plug]
                if apply_button.draw(screen):
                    for audio in audio_clips:
                        if audio.track == selected_track:
                            audio.reapply()
                draw_text(screen, "Reverb Room", font35, (X_OFF + 90, MENU_OFFSET + 20), WHITE)
                
                if reverb_values[selected_track][-1] > reverb_values[selected_track][-2]:
                    draw_text(screen, "Warning: Tail length in seconds must be greater than or equal to the delay length in seconds!", font, (X_OFF + 90, MENU_OFFSET + 420), WHITE)

            if my_plug == "Delay":
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                s.fill((0, 128, 132, 128))
                screen.blit(s, (X_OFF, MENU_OFFSET-30))
                names = ["Mix", f"Delay ({round(delay_values[selected_track][1] * 2000)} ms)", "Spread"]
                for i, name in enumerate(names):
                    x = X_OFF + 45 + (i * 80 * 7/3)
                    draw_text(screen, name, font, (x - 20, MENU_OFFSET + 140), WHITE)
                for i, knob in enumerate(delay_knobs[selected_track]):
                    knob.draw(screen)
                    delay_values[selected_track][delay_knobs[selected_track].index(knob)] = knob.value
                    x = X_OFF + 45 + (i * 80 * 7/3)
                    draw_text(screen, f"{round((knob.value)*100, 1)}%", font35, (x - 20, MENU_OFFSET + 170), WHITE)
                if apply_button.draw(screen):
                    for audio in audio_clips:
                        if audio.track == selected_track:
                            audio.reapply()
                on_button.text = "On" if studio_effects[my_plug] else "Off"
                if on_button.draw(screen):
                    studio_effects[my_plug] = not studio_effects[my_plug]
                draw_text(screen, "Dusted Delay Filter", font35, (X_OFF + 90, MENU_OFFSET + 20), WHITE)
                draw_delay_visualizer(screen, X_OFF + 50, MENU_OFFSET + 250, WIDTH - X_OFF - 100, 150, delay_values[selected_track])
            if my_plug == "Compressor":
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                s.fill((227, 223, 0, 128))
                screen.blit(s, (X_OFF, MENU_OFFSET-30))
                names = ["Threshold", "Ratio", f"Attack ({round(lerp(compressor_values[selected_track][2], 0.1, 200), 1)} ms)", f"Release ({round(lerp(compressor_values[selected_track][3], 10, 2000), 1)} ms)"]
                for i, name in enumerate(names):
                    x = X_OFF + 45 + (i * 80 * 7/4)
                    draw_text(screen, name, font, (x - 20, MENU_OFFSET + 140), WHITE)
                for i, knob in enumerate(compressor_knobs[selected_track]):
                    knob.draw(screen)
                    compressor_values[selected_track][compressor_knobs[selected_track].index(knob)] = knob.value
                    x = X_OFF + 45 + (i * 80 * 7/4)
                    draw_text(screen, f"{round((knob.value)*100, 1)}%", font35, (x - 20, MENU_OFFSET + 170), WHITE)

                if apply_button.draw(screen):
                    for audio in audio_clips:
                        if audio.track == selected_track:
                            audio.reapply()
                on_button.text = "On" if studio_effects[my_plug] else "Off"
                if on_button.draw(screen):
                    studio_effects[my_plug] = not studio_effects[my_plug]

                draw_text(screen, "Studio Compressor", font35, (X_OFF + 90, MENU_OFFSET + 20), WHITE)
                draw_compressor_visualizer(screen, X_OFF + 50, MENU_OFFSET + 250, WIDTH - X_OFF - 100, 150, compressor_values[selected_track])

            if my_plug == "Gate":
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                s.fill((17, 115, 78, 128))
                screen.blit(s, (X_OFF, MENU_OFFSET-30))
                names = [f"Gain ({round(lerp(gate_values[selected_track][0], -60, 60))} db)", f"Threshold ({round(lerp(gate_values[selected_track][1], -60, 0))} db)"]
                for i, name in enumerate(names):
                    x = X_OFF + 45 + (i * 80 * 7/2)
                    draw_text(screen, name, font, (x - 20, MENU_OFFSET + 140), WHITE)
                for i, knob in enumerate(gate_knobs[selected_track]):
                    knob.draw(screen)
                    gate_values[selected_track][gate_knobs[selected_track].index(knob)] = knob.value
                    x = X_OFF + 45 + (i * 80 * 7/2)
                    draw_text(screen, f"{round((knob.value)*100, 1)}%", font35, (x - 20, MENU_OFFSET + 170), WHITE)

                if apply_button.draw(screen):
                    for audio in audio_clips:
                        if audio.track == selected_track:
                            audio.reapply()
                on_button.text = "On" if studio_effects[my_plug] else "Off"
                if on_button.draw(screen):
                    studio_effects[my_plug] = not studio_effects[my_plug]

                draw_text(screen, "Gategain", font35, (X_OFF + 90, MENU_OFFSET + 20), WHITE)
                draw_gain_threshold_visualizer(screen, X_OFF + 50, MENU_OFFSET + 250, WIDTH - X_OFF - 100, 150, gate_values[selected_track])

            if my_plug == "Chorus":
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                s.fill((116, 184, 95, 128))
                screen.blit(s, (X_OFF, MENU_OFFSET-30))
                names = ["LFO Speed", "Depth", f"Wide ({round(lerp(chorus_values[selected_track][2], 1, 30), 2)} ms)", "Feedback", "Mix"]
                for i, name in enumerate(names):
                    x = X_OFF + 45 + (i * 80 * 7/5)
                    draw_text(screen, name, font, (x - 20, MENU_OFFSET + 140), WHITE)
                for i, knob in enumerate(chorus_knobs[selected_track]):
                    knob.draw(screen)
                    chorus_values[selected_track][chorus_knobs[selected_track].index(knob)] = knob.value
                    x = X_OFF + 45 + (i * 80 * 7/5)
                    draw_text(screen, f"{round((knob.value)*100, 1)}%", font35, (x - 20, MENU_OFFSET + 170), WHITE)


                draw_chorus_visualizer(screen, X_OFF + 50, MENU_OFFSET + 250, WIDTH - X_OFF - 100, 150, chorus_values[selected_track], time=pygame.time.get_ticks() / 1000)
                
            

                on_button.text = "On" if studio_effects[my_plug] else "Off"
                if on_button.draw(screen):
                    studio_effects[my_plug] = not studio_effects[my_plug]
                if apply_button.draw(screen):
                    for audio in audio_clips:
                        if audio.track == selected_track:
                            audio.reapply()
                draw_text(screen, "Chorus Creator", font35, (X_OFF + 90, MENU_OFFSET + 20), WHITE)
                

            if my_plug != "None":
                draw_text(screen, f"Track {selected_track+1}", font, (X_OFF + 20, MENU_OFFSET), WHITE)

        if plugin_button.draw(screen):
            plugins = not plugins
        screen.blit(plugin_img, (760, 40))

        if not show_paths:
            if last_clicked == None:
                for slider in sliders:
                    col = colors[slider.number % 10] if not muted[slider.number] else (200, 200, 200)
                    pygame.draw.rect(screen, col, (slider.x-10, slider.y-25, X_OFF-(slider.x-10)-5, 22))
                    draw_text(screen, f"Track {slider.number+1}", font, (slider.x, slider.y-22), BLACK)
                    if len(automations[slider.number]["volume"]) == 0:
                        slider.draw(screen)
                        slider.size = mixer_volumes[slider.number]
                        # mixer_volumes[slider.number].set_volume(slider.get_value())
                    else:
                        draw_text(screen, f"(AUTO)", font, (slider.x, slider.y), WHITE)

                for button in mute_buttons:
                    if button.draw(screen):
                        muted[button.number] = not muted[button.number]
                    button.original_col = (200,200,200) if muted[button.number] else BUTTON_COL

                for button in solo_buttons:
                    button.original_col = (255,174,20) if soloed[button.number] else BUTTON_COL
                    if button.draw(screen):
                        soloed[button.number] = not soloed[button.number]
                        
                        for channel in range(CHANNELS):
                            if not soloed[channel]:
                                muted[channel] = True
                            else:
                                muted[channel] = False
                        if all(muted):
                            muted = [False]*CHANNELS
                            soloed = [False]*CHANNELS
            else:
                draw_text(screen, f"Pitch", font, (55, MENU_OFFSET+5), WHITE)
                draw_text(screen, f"{round(last_clicked.display_semitones, 2)} st", font, (55, MENU_OFFSET+55), WHITE)
                if pitch_plus.draw(screen): last_clicked.pitch(1)
                if pitch_minus.draw(screen): last_clicked.pitch(-1)
                if pitch_fplus.draw(screen): last_clicked.pitch(0.25)
                if pitch_fminus.draw(screen): last_clicked.pitch(-0.25)

                if speed_plus.draw(screen): 
                    last_clicked.timeStretch(1.05)
                if speed_minus.draw(screen): 
                    last_clicked.timeStretch(0.95)
                if speed_fplus.draw(screen): 
                    last_clicked.timeStretch(1.01)
                if speed_fminus.draw(screen): 
                    last_clicked.timeStretch(0.99)

                draw_text(screen, f"Speed", font, (55, MENU_OFFSET+115), WHITE)
                draw_text(screen, f"x {round(float(last_clicked.speed_rate), 2)}", font, (55, MENU_OFFSET+165), WHITE)

        else:
            y = MENU_OFFSET+10 + paths_y
            for path, button in paths_to_save.copy():
                if y >= MENU_OFFSET-10:
                    name = path.split("/")[-1]
                    button.rect.y = y+15
                    if len(name) > 17:
                        name = name[:15] + "..."
                    if button.draw(screen):
                        # try:
                            current_time = time.time()
                            if not insert_mode:
                                preview_channel.stop()
                                preview_channel.play(pygame.mixer.Sound(path))
                            else:
                                # split()
                                spawn_clip(path)

                            last_click_time = current_time
                        # except FileNotFoundError:
                            # button.text = "File Not Found"
                
                    draw_text(screen, name, font, (5, y-3), WHITE)
                    draw_waveform(path, screen, BLACK, (5, y+15), (X_OFF-10, 35))
                    
                y += 55

        if stretch_mode:
            screen.blit(stretch_mouse_img, (mx, my))

    if mixer:
        screen.fill(BLACK)
        x_set = X_OFF*0.5 + 200
        pygame.draw.rect(screen, "#3d4e55", (X_OFF*0.5, MENU_OFFSET*0.5, WIDTH-X_OFF, HEIGHT-MENU_OFFSET))
        for i in range(CHANNELS):
            r = pygame.Rect(x_set + (channel_width * i), MENU_OFFSET*0.5 + 40, channel_width+4, HEIGHT-MENU_OFFSET-80)
            pygame.draw.rect(screen, "#202426", r)
            pygame.draw.rect(screen, "#919191", r, 4)
            draw_text(screen, f"{i+1}", font, (r.x+35, r.y-20), WHITE)
            
            # Draw vertical slider
            slider_width = channel_width * 0.5
            slider_x = r.x + (channel_width - slider_width) / 2 + 3
            slider_height = (r.height - 20) * 0.7
            slider_y = r.y + r.height * (0.3)
            
            # Draw slider background
            pygame.draw.rect(screen, (150, 150, 150), (slider_x + slider_width*0.25, slider_y, slider_width * 0.5, slider_height), border_radius=5)
            
            # Draw slider fill
            fill_height = int(mixer_volumes[i] * slider_height)
            pygame.draw.rect(screen, GREEN, (slider_x + slider_width*0.25, slider_y + slider_height - fill_height, slider_width * 0.5, fill_height), border_radius=5)
            
            # Draw handle
            handle_y = slider_y + slider_height - fill_height
            handle_x = slider_x + slider_width / 2
            pygame.draw.rect(screen, (220,220,220), pygame.Rect(handle_x-16, handle_y-8, 32, 16))
            # pygame.draw.circle(screen, (0, 0, 0), (int(handle_x), int(handle_y)), int(slider_width / 2))
            
            # Handle slider interaction
            mouse_pos = pygame.mouse.get_pos()
            if r.collidepoint(mouse_pos) and pygame.mouse.get_pressed()[0] and mouse_pos[1] >= slider_y-10:
                relative_y = mouse_pos[1] - slider_y
                mixer_volumes[i] = max(0, min(1, 1 - (relative_y / slider_height)))

            pan_knobs[i].draw(screen)
            mixer_pan[i] = pan_knobs[i].value

            if plugin_mixer[i].draw(screen):
                mixer = False
                plugins = True
                selected_track = i

        r = pygame.Rect(x_set - 150, MENU_OFFSET*0.5 + 30, 100+4, HEIGHT-MENU_OFFSET-60)
        pygame.draw.rect(screen, "#202426", r)
        pygame.draw.rect(screen, "#919191", r, 4)
        draw_text(screen, f"Master", font, (r.x+27, r.y-20), WHITE)
        
        # Draw vertical slider
        slider_width = channel_width * 0.65
        slider_x = r.x + (channel_width - slider_width) / 2 + 15
        slider_height = (r.height - 20) * 0.7
        slider_y = r.y + r.height * (0.3)
        
        # Draw slider background
        pygame.draw.rect(screen, (150, 150, 150), (slider_x + slider_width*0.25, slider_y, slider_width * 0.5, slider_height), border_radius=5)
        
        # Draw slider fill
        fill_height = int(master_volume * slider_height)
        pygame.draw.rect(screen, GREEN, (slider_x + slider_width*0.25, slider_y + slider_height - fill_height, slider_width * 0.5, fill_height), border_radius=5)
        
        # Draw handle
        handle_y = slider_y + slider_height - fill_height
        handle_x = slider_x + slider_width / 2
        pygame.draw.rect(screen, (220,220,220), pygame.Rect(handle_x-16, handle_y-8, 32, 16))
        # pygame.draw.circle(screen, (0, 0, 0), (int(handle_x), int(handle_y)), int(slider_width / 2))
        
        # Handle slider interaction
        mouse_pos = pygame.mouse.get_pos()
        if r.collidepoint(mouse_pos) and pygame.mouse.get_pressed()[0] and mouse_pos[1] >= slider_y-10:
            relative_y = mouse_pos[1] - slider_y
            master_volume = max(0, min(1, 1 - (relative_y / slider_height)))

        master_pan_knob.draw(screen)
        master_pan = master_pan_knob.value

        
        RS["oi"] += 1
        if RS["oi"] > 50:
            RS["oi"] = 0
            big_ahh_cache = {}


        if mixer_apply_button.draw(screen):
                for audio in audio_clips:
                    audio.reapply()
        if leave_button.draw(screen):
            mixer = False

    dmx = 0
    for event in pygame.event.get():
        # for slider in sliders:
        #     slider.handle_event(event)

        if my_plug == "EQ": 
            for knob in eq_knobs[selected_track]: knob.handle_event(event)
        if my_plug == "Reverb":
            for knob in reverb_knobs[selected_track]: knob.handle_event(event)
        if my_plug == "Delay":
            for knob in delay_knobs[selected_track]: knob.handle_event(event)
        if my_plug == "Compressor":
            for knob in compressor_knobs[selected_track]: knob.handle_event(event)
        if my_plug == "Gate":
            for knob in gate_knobs[selected_track]: knob.handle_event(event)
        if my_plug == "Chorus":
            for knob in chorus_knobs[selected_track]: knob.handle_event(event)
        if my_plug == 'None':
            metro_slider.handle_event(event)
            aux_channel.set_volume(metro_slider.get_value())
        if mixer:
            for i in range(CHANNELS):
                pan_knobs[i].handle_event(event)
            master_pan_knob.handle_event(event)
        # pan_slider.handle_event(event)
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEWHEEL:
            if not mixer:
                dmx = -event.x*10
                if left_sidebar.collidepoint((mx, my)):
                    paths_y += event.y * 10
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                preview_channel.stop()
                if seq.on:
                    seq.stop()
                else:
                    seq.run()

            if event.key == pygame.K_q:
                if many_sel:
                    many_sel = False
                    anything_grabbed = False
                    for audio in audio_clips:
                        audio.grabbed = False
                        audio.single_selected = True
            if event.key == pygame.K_c:
                if event.mod & pygame.KMOD_META:
                    consolidate()
                else:
                    ads = []
                    minx = float("inf")
                    mint = float("inf")
                    if many_sel:
                        for audio in audio_clips:
                            if audio.grabbed and (not audio.single_selected):
                                ads.append(audio)
                                minx = min(minx, audio.x-acc)
                                mint = min(mint, audio.track)
                        copied_audio = {"full": ads, "min x": minx, "min track": mint}
            if event.key == pygame.K_v:
                if event.mod & pygame.KMOD_META:
                    if last_clicked != None:
                        for i in range(1, maxa):
                            Audio(last_clicked.path, last_clicked.rect.x + (LPS//maxa)*i, last_clicked.rect.y)
                else:
                    for audio in copied_audio["full"]:
                        new_path = f"{"".join(audio.original_path.split(".")[:-1])} | COPY{RANDOM_CONSTANT}.wav"
                        RANDOM_CONSTANT += 1
                        copy_audio(audio.original_path, new_path)
                        Audio(new_path, seq.rect.x+(audio.x-copied_audio['min x'])-acc, ((selected_track+(audio.track-copied_audio["min track"]))*TRACK_HEIGHT+MENU_OFFSET))
            
            if event.key == pygame.K_r:
                if event.mod & pygame.KMOD_META:
                    if isinstance(last_clicked, Audio):
                        last_clicked.reverse()
                else:
                    record()

            if event.key == pygame.K_g:
                if last_clicked != None:
                    last_clicked.grabbed = False
                    last_clicked = None
                    anything_grabbed = False

            if event.key == pygame.K_a:
                if event.mod & pygame.KMOD_META:
                    for audio in audio_clips:
                            anything_grabbed = True
                            audio.grabbed = True
                            audio.single_selected = False
                            many_sel = True
                    not_pressed_before = True
                    main_point = [None, None]
                # else:
                #     stretch_mode = False
                #     auto = not auto
                #     pygame.mouse.set_visible(True)

            if event.key == pygame.K_m:
                if (not auto) and my_plug=="None":
                    if stretch_mode:
                        stretch_mode = False
                        pygame.mouse.set_visible(True)
                    else:
                        stretch_mode = True
                        pygame.mouse.set_visible(False)

            if event.key == pygame.K_e:
                mixer = not mixer
            
            if event.key == pygame.K_f:
                show_paths = not show_paths
                paths_y = 0
            
            if event.key == pygame.K_i:
                insert_mode = not insert_mode 

     
            if event.key == pygame.K_LEFT:
                if event.mod & pygame.KMOD_META: seq.rect.x -= LPS
                else: seq.rect.x -= (LPS // maxa)
                reset_clips_playing()
            if event.key == pygame.K_RIGHT: 
                if event.mod & pygame.KMOD_META: seq.rect.x += LPS
                else: seq.rect.x += (LPS // maxa)               
                reset_clips_playing()

            if event.key == pygame.K_t:
                if event.mod & pygame.KMOD_META:
                    old_bpm = seq.tempo

            if event.key == pygame.K_u:
                load_audio()

            if event.key == pygame.K_x:
                old_lps = LPS
                LPS = int(min(LPS+(LPS/80 * 20), 800))
                for audio in audio_clips:
                    audio.x = ((LPS*audio.x) + (acc+X_OFF)*(old_lps-LPS)) / old_lps
                seq.stop()

            if event.key == pygame.K_z:
                old_lps = LPS
                LPS = int(max(LPS-(LPS/80 * 20), 50))
                for audio in audio_clips:
                    audio.x = ((LPS*audio.x) + (acc+X_OFF)*(old_lps-LPS)) / old_lps
                seq.stop()

            if event.key == pygame.K_TAB:
                if auto:
                    fine = True
                    mp = pygame.mouse.get_pos()
                    if MENU_OFFSET <= mp[1] and X_OFF <= mp[0]:
                        track = (mp[1]-MENU_OFFSET) // TRACK_HEIGHT

                        for effect in automations[track]:
                            for point in automations[track][effect]:
                                if point.clicked():
                                    fine = False
                                    
                        if fine:
                            AutoPoint(mp[0], mp[1], track, chosen_effect)
                    
            # if event.key == pygame.K_BACKSPACE:
            #     if many_sel:
            #         many_sel = False
            #         anything_grabbed = False
            #         copyclip = audio_clips
            #         for audio in copyclip:
            #             if audio.grabbed:
            #                 audio_clips.remove(audio)
            

    acc += dmx
    pygame.display.flip()

stream.stop()
stream.close()
pygame.quit()
for filename in os.listdir("sounds"):
    file_path = os.path.join("sounds", filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
for filename in os.listdir("temps"):
    file_path = os.path.join("temps", filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
output("Closed successfully!")