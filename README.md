# DRUMBER

Takes random-ish 300ms samples from an input mp3 or wav and assigns 15 of them to each drum channel. Shape and cycle through them using euclidean divisions and randomness, along with 4 LFOs and 3 assignable FX.
<img width="1813" height="1286" alt="image" src="https://github.com/user-attachments/assets/fd6562f7-0aca-470b-a3aa-fb792b82ae1c" />

## Features:

### Sample Engine

- Auto-Extraction: Loads any .wav or .mp3 and uses librosa to automatically slice transients and categorize them by sonic profile (Kick, Snare, Hats, Toms, Crash).
- Cycle & Rescan: Cycle through 15 curated samples per channel. Hold the Cycle button to get 15 brand new samples for that specific drum, available during playback.
- Sample Trimming: Dual-handle sliders on each channel to visually set sample start and stop times (0–300ms) of the drum hits.

### Sequencing & Performance

- Dynamic Grid: 8 channels over 8, 16, or 32 steps. Expand the grid and it automatically loops your beat into the newly created space. Left-click/drag to paint beats, right-click to erase.
- Burst Mode: Hold the "BURST" button to multiply the tempo (from 0.5x up to 8x) for real-time stutters and fills. When released, it drops you back onto the underlying master clock.
- Euclidean Generator: Generate complex polyrhythms instantly. Includes weighted "Dice" randomizers per-channel or globally that use a custom euclidean probability to generate unique beats.
- Pattern Snapshots: 4 switchable patterns that save the sequence with a snapshot of each channel setting (volume, pitch, FX sends, etc.).

### Modulation & FX Rack

- LFO Mod Matrix: 3 independent LFOs (Sine, Triangle, Square, Random) that can go as slow as 4 BPM or sync to note divisions. Middle-click any knob, slider, or grid step in the entire UI to target it with an LFO. Click "X" beside LFO to disengage.
- Studio FX Chain: 3 serial master FX buses (FX1 -> FX2 -> FX3) powered by Spotify's pedalboard (Reverb, Delay, Distortion, Compressor, Resonant LPF).

### Workflow & Quality of Life

- Advanced Undo System: Global 24-step Undo button, plus targeted independent Undos (Right-click any global randomize button to revert just what that button changed).
- Ctrl-Hold Tweaking: Hold Ctrl while moving any slider/knob to instantly mirror that adjustment across all unlocked channels simultaneously.
- Channel Locks: Lock individual channels to protect them from global randomizers and Ctrl-group tweaks.
- Live Recording: Hit "Record" to record and save directly to a .wav file on your hard drive.

# INSTALL AND RUN

Install python, then run:

`
pip install numpy librosa sounddevice customtkinter scipy pedalboard
`

If you are on macOS or Linux, you might also need to install the underlying PortAudio system library for sounddevice to work, and the Tkinter framework for the GUI:
Ubuntu/Debian: 

`sudo apt-get install libportaudio2 python3-tk`

macOS (using Homebrew): 

`brew install portaudio python-tk`

Run with:

`python3 drumber10.py`

# UPDATES
8.3 v2: 

DSP & Audio Engine Enhancements
- Stutter Engine: Replaced the "Burst" (BPM multiplier) system with a real-time Audio Buffer Stutter. It captures a slice of the master output and loops it with 1ms crossfades to prevent clicks, while keeping the background sequencer perfectly in sync.
- Reverse Playback: Added a per-channel Reverse (R) toggle that flips audio samples for backwards playback without affecting timing.
- Chorus Effect: Integrated a global Chorus effect into the FX rack options.
- Triplet Stuttering: Expanded stutter divisions to include valid triplet values (e.g., 1/4T, 1/8T, 1/16T, 1/32T).
- BPM Auto-Detection: The engine now automatically analyzes loaded files to estimate their native tempo and updates the global BPM (toggleable in Options).

UI & Visualization
- Playback Head: Added a visual "glow" indicator in the sequencer grid that highlights the current active step across all channels during playback.
- Resolution Increase: Updated the initial window geometry to 1450x1075 to better accommodate the mixer and sequencer layout.
- Keyboard Shortcuts: Enabled Spacebar to toggle Play/Stop (safeguarded against firing while typing in dialogs).
- Visual Feedback: Enhanced the "Reverse" and "Stutter" buttons with color-coded states for better visibility.

Workflow & Control Logic
- Sample Swapping Context Menu: Added a right-click menu to Channel names that allows users to "Send sample to" another channel. This performs a physical sample swap between tracks and automatically selects the new index for immediate playback.
- Stutter Mute Modes: Implemented dual stutter behavior: Left-click allows background audio to play through; Right-click mutes the background for "solo" stutter effects.
- Enhanced LFO Mapping:
        Fixed a crash when mapping Euclidean sliders to LFOs.
        Enabled 🎲 (Dice) buttons as LFO targets, allowing LFOs to trigger random pattern generation.
        Added middle-click LFO support for the Stutter button and Stutter division knob.
- Mass Slider Control: Fixed the "Ctrl + Drag" logic to ensure the Cycle slider now correctly adjusts all channels simultaneously, matching the behavior of other sliders.

Randomization & Resets
- Fade Randomization: Replaced global Pitch randomization with a Fade (Fad) randomizer for creating evolving rhythmic envelopes.
- Double-Click Resets: Double-clicking the "Fad" randomizer now resets all channel fades to default values (0ms In / 100% Out).
- Options Menu Expansion: Added toggles for Stutter background behavior and Auto-BPM detection.
