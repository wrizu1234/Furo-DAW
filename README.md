# Furo-DAW
Furo is a digital audio workstation (DAW) coded entirely in Python that can manipulate audio files in order to create whatever you want! From combining clips, to reversing, splitting, and quick insertion, FURO is THE simple yet effective DAW. Use the mixer to control volume and pan, and use the plugin feature to add reverb, delay, EQ, compression, chorus, noise gates, and much more effects! It's not about what this DAW can do-- it's about who YOU will BECOME.

# Keyboard Controls

## Playback & Recording

### Spacebar  
**Play or Stop the Project**  
Starts or stops the main audio sequence. Also stops any preview sound that's currently playing.

### R  
**Record Audio**  
Begins recording a new audio clip.

### Command (⌘) + R  
**Reverse Audio**  
Reverses the last clicked audio clip so it plays backward.

---

## Clip Selection & Editing

### Q  
**Deselect All Clips**  
Disables multi-selection mode and resets all clips to be individually selectable.

### Command (⌘) + A  
**Select All Clips**  
Grabs all audio clips so they can be moved or edited together.

### G  
**Release Selected Clip**  
Ungrabs the last clicked audio clip and resets dragging state.

---

## Copying & Pasting Audio

### C  
**Copy Selected Clips**  
Copies all currently selected audio clips (except the first one clicked) and stores their relative position.

### Command (⌘) + C  
**Consolidate**  
Combines selected clips into one (if this function is implemented).

### V  
**Paste Copied Clips**  
Pastes the copied audio clips back into the timeline at a position relative to the sequence.

### Command (⌘) + V  
**Duplicate Last Clip in Pattern**  
Creates a series of duplicates of the last clicked clip, laid out in a rhythmic pattern.

---

## Tools & Views

### M  
**Toggle Stretch Mode**  
Enables or disables stretch mode for time-stretching audio. Also hides the mouse cursor while active.

### E  
**Toggle Mixer View**  
Turns the mixer on or off for adjusting audio parameters like volume and panning.

### F  
**Show or Hide File Paths**  
Toggles visibility of file paths or names below each clip.

### I  
**Toggle Insert Mode**  
Switches insert mode on or off, affecting how new clips are placed.

---

## Navigation & Zoom

### Left Arrow (←) or Right Arrow (→)  
**Scroll Timeline**  
Moves the sequence view left or right.

### Command (⌘) + Left or Right Arrow  
**Jump in Timeline**  
Scrolls by a larger amount for faster navigation.

### X  
**Zoom In**  
Increases the zoom level of the timeline, making clips appear wider.

### Z  
**Zoom Out**  
Decreases the zoom level, showing more of the timeline at once.

---

## Miscellaneous

### U  
**Import Audio File**  
Opens a file picker to load a new audio clip into the project.

### Command (⌘) + T  
**Store Tempo Setting**  
Saves the current BPM (tempo) value internally (no visible change).

