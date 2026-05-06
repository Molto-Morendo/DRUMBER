import os, copy, math, random, threading, time, warnings
from collections import deque
import numpy as np
import librosa
import sounddevice as sd
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import scipy.io.wavfile as wavfile
from pedalboard import Pedalboard, Reverb, Delay, Distortion, Compressor, LowpassFilter
try:
    from pedalboard import LadderFilter
    HAS_LADDER = True
except ImportError: HAS_LADDER = False

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SAMPLE_RATE, BLOCK_SIZE = 44100, 1024
CHANNELS =["Kick", "Snare", "Closed Hat", "Open Hat", "Hi Tom", "Low Tom", "Crash", "Random"]
ROW_COLORS =["#E74C3C", "#F39C12", "#3498DB", "#2980B9", "#9B59B6", "#8E44AD", "#1ABC9C", "#BDC3C7"]
APP_FONT, APP_FONT_BOLD = ("Trebuchet MS", 12), ("Trebuchet MS", 12, "bold")

FX_DEFS = {
    "Reverb":[("Size",0.0,1.0,0.5), ("Length",0.0,1.0,0.5), ("Mix",0.0,1.0,0.0)],
    "Delay":[("Time",0.0,2.0,0.3), ("Repeats",0.0,1.0,0.4), ("Mix",0.0,1.0,0.0)],
    "Distortion":[("Amount",0.0,1.0,0.5), ("Mix",0.0,1.0,1.0), ("",0.0,1.0,0.0)],
    "Compressor":[("Thresh",0.0,1.0,0.1), ("Ratio",1.0,20.0,4.0), ("Gain",0.0,1.0,0.05)],
    "Resonant LPF":[("Cutoff",20.0,20000.0,20000.0), ("Res",0.0,1.0,0.0), ("",0.0,1.0,0.0)]
}

# --- CUSTOM WIDGETS ---
class CTkKnob(ctk.CTkFrame):
    def __init__(self, master, width=40, height=40, from_=0.0, to=1.0, command=None, fg_color="transparent", **kwargs):
        super().__init__(master, width=width, height=height, fg_color=fg_color, **kwargs)
        self.min_val, self.max_val, self.command, self.value = from_, to, command, from_
        self.is_dragging = False
        self.canvas = ctk.CTkCanvas(self, width=width, height=height, bg="#1A1A1A", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<B1-Motion>", self._on_drag); self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self._last_y = 0; self.draw()
    def configure_range(self, from_, to): self.min_val, self.max_val = from_, to; self.set(self.value)
    def set(self, value): self.value = max(self.min_val, min(self.max_val, value)); self.draw()
    def get(self): return self.value
    def _on_press(self, event): self._last_y = event.y; self.is_dragging = True
    def _on_release(self, event): self.is_dragging = False
    def _on_drag(self, event):
        dy = self._last_y - event.y; self._last_y = event.y
        new_val = max(self.min_val, min(self.max_val, self.value + ((dy / 100.0) * (self.max_val - self.min_val))))
        if new_val != self.value:
            self.value = new_val; self.draw()
            if self.command: self.command(self.value)
    def draw(self):
        self.canvas.delete("all")
        w, h = int(self.canvas.cget("width")), int(self.canvas.cget("height")); cx, cy, r = w/2, h/2, min(w, h)/2 - 6 
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=225, extent=-270, style="arc", outline="#333333", width=4)
        pct = (self.value - self.min_val) / (self.max_val - self.min_val) if self.max_val > self.min_val else 0
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=225, extent=-270*pct, style="arc", outline="#3B8ED0", width=4)
        ang = math.radians(225 + -270*pct)
        self.canvas.create_line(cx + (r-6)*math.cos(ang), cy - (r-6)*math.sin(ang), cx + r*math.cos(ang), cy - r*math.sin(ang), fill="#AAAAAA", width=2)

class CTkRangeSlider(ctk.CTkFrame):
    def __init__(self, master, width=130, height=20, from_=0, to=300, command=None, **kwargs):
        super().__init__(master, width=width, height=height, fg_color="transparent", **kwargs)
        self.min_val, self.max_val, self.val_start, self.val_end, self.command = from_, to, from_, 200, command
        self.is_dragging = False
        self.canvas = ctk.CTkCanvas(self, width=width, height=height, bg="#1A1A1A", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press); self.canvas.bind("<B1-Motion>", self._on_drag); self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.active_handle = None; self.draw()
    def set(self, v_start, v_end):
        self.val_start, self.val_end = max(self.min_val, min(self.max_val, v_start)), max(self.val_start + 1, min(self.max_val, v_end))
        self.draw()
    def get(self): return self.val_start, self.val_end
    def _val_to_x(self, val, w): return 10 + ((val - self.min_val) / (self.max_val - self.min_val)) * (w - 20)
    def _x_to_val(self, x, w): return max(self.min_val, min(self.max_val, self.min_val + ((x - 10) / (w - 20)) * (self.max_val - self.min_val)))
    def _on_press(self, event):
        self.is_dragging = True; w = int(self.canvas.cget("width"))
        self.active_handle = 'start' if abs(event.x - self._val_to_x(self.val_start, w)) < abs(event.x - self._val_to_x(self.val_end, w)) else 'end'
    def _on_release(self, event): self.is_dragging = False
    def _on_drag(self, event):
        new_val = self._x_to_val(event.x, int(self.canvas.cget("width")))
        if self.active_handle == 'start': self.val_start = min(new_val, self.val_end - 5)
        else: self.val_end = max(new_val, self.val_start + 5)
        self.draw(); self.command(self.val_start, self.val_end) if self.command else None
    def draw(self):
        self.canvas.delete("all")
        w, cy = int(self.canvas.cget("width")), int(self.canvas.cget("height")) / 2; xs, xe = self._val_to_x(self.val_start, w), self._val_to_x(self.val_end, w)
        self.canvas.create_line(10, cy, w-10, cy, fill="#333333", width=4, capstyle="round"); self.canvas.create_line(xs, cy, xe, cy, fill="#3B8ED0", width=4, capstyle="round")
        self.canvas.create_oval(xs-5, cy-5, xs+5, cy+5, fill="#DDDDDD", outline="#888888"); self.canvas.create_oval(xe-5, cy-5, xe+5, cy+5, fill="#DDDDDD", outline="#888888")

class DraggableBPM(ctk.CTkFrame):
    def __init__(self, master, app_ref, default=120, **kwargs):
        super().__init__(master, fg_color="#1F1F1F", corner_radius=6, border_width=1, border_color="#333333", **kwargs)
        self.app = app_ref
        self.lbl = ctk.CTkLabel(self, text=f"BPM\n{int(default)}", font=APP_FONT_BOLD, text_color="#2ECC71", width=50); self.lbl.pack(padx=8, pady=2)
        self.lbl.bind("<B1-Motion>", self._on_drag); self.lbl.bind("<ButtonPress-1>", lambda e: setattr(self, '_last_y', e.y_root)); self.lbl.bind("<Double-Button-1>", self._on_double)
        self._last_y = 0
    def set_val(self, val):
        self.app.engine.bpm = max(40, min(300, int(val))); self.lbl.configure(text=f"BPM\n{int(self.app.engine.bpm)}")
    def _on_drag(self, e):
        dy = self._last_y - e.y_root
        if abs(dy) > 2:
            self.app.save_state(); self.set_val(self.app.engine.bpm + (dy // 3)); self._last_y = e.y_root
    def _on_double(self, e):
        res = ctk.CTkInputDialog(text="Enter BPM (40-300):", title="BPM Entry").get_input()
        if res:
            try: self.app.save_state(); self.set_val(int(res))
            except ValueError: pass

# --- DSP ENGINE & LFO ---
class LFO:
    def __init__(self, color):
        self.color, self.shape, self.sync = color, "Sine", False
        self.rate_hz, self.rate_sync, self.depth = 1.0, "1/4", 0.5
        self.target_id, self.target_name, self.phase, self.val, self.last_snh = None, None, 0.0, 0.0, 0.0
    def step(self, delta_time, bpm, trigger_queue):
        prev_val = self.val
        if self.sync:
            mults = {"1/16":4, "1/8":2, "1/4":1, "1/2":0.5, "1 Bar":0.25, "2 Bar":0.125, "4 Bar":0.0625, "8 Bar":0.03125}
            freq = (max(1.0, bpm) / 60.0) * mults.get(self.rate_sync, 1)
        else: freq = self.rate_hz
        self.phase += freq * delta_time
        if self.phase >= 1.0: self.phase -= 1.0; self.last_snh = random.uniform(-1.0, 1.0)
        p = self.phase
        if self.shape == "Sine": self.val = math.sin(2 * math.pi * p)
        elif self.shape == "Triangle": self.val = 2.0 * abs(2.0 * (p - 0.5)) - 1.0
        elif self.shape == "Square": self.val = 1.0 if p < 0.5 else -1.0
        elif self.shape == "Random": self.val = self.last_snh
        
        # Detect Rising edge through zero for Gate Targets (Buttons)
        if self.target_id:
            if self.target_id.startswith("gl_rand:") or self.target_id.endswith(":cyc") or self.target_id.endswith(":trigger"):
                if prev_val <= 0 and self.val > 0:
                    trigger_queue.append(self.target_id)

class GlobalFXBus:
    def __init__(self, fx_type):
        self.fx_type, self.p1, self.p2, self.p3, self.m_p1, self.m_p2, self.m_p3 = fx_type, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.plugin, self.lock = None, threading.Lock(); self.set_type(fx_type)
    def set_type(self, fx_type):
        with self.lock:
            self.fx_type = fx_type
            if fx_type == "Reverb": self.plugin = Reverb()
            elif fx_type == "Delay": self.plugin = Delay()
            elif fx_type == "Distortion": self.plugin = Distortion()
            elif fx_type == "Compressor": self.plugin = Compressor()
            elif fx_type == "Resonant LPF": self.plugin = LadderFilter(mode=LadderFilter.Mode.LPF12) if HAS_LADDER else LowpassFilter()
            self.p1, self.p2, self.p3 = FX_DEFS[fx_type][0][3], FX_DEFS[fx_type][1][3], FX_DEFS[fx_type][2][3]
            self.m_p1, self.m_p2, self.m_p3 = self.p1, self.p2, self.p3; self._apply_params()
    def set_param(self, idx, val):
        with self.lock:
            if idx == 1: self.p1 = val
            elif idx == 2: self.p2 = val
            elif idx == 3: self.p3 = val
    def _apply_params(self):
        if not self.plugin: return
        try:
            if self.fx_type == "Reverb":
                self.plugin.room_size, self.plugin.damping = self.m_p1, 1.0 - self.m_p2
                self.plugin.wet_level, self.plugin.dry_level = self.m_p3, 1.0 - self.m_p3
            elif self.fx_type == "Delay": self.plugin.delay_seconds, self.plugin.feedback, self.plugin.mix = self.m_p1, self.m_p2, self.m_p3
            elif self.fx_type == "Distortion": self.plugin.drive_db = self.m_p1 * 40.0
            elif self.fx_type == "Compressor": self.plugin.threshold_db, self.plugin.ratio = -(self.m_p1 * 60.0), self.m_p2
            elif self.fx_type == "Resonant LPF":
                self.plugin.cutoff_hz = max(20.0, self.m_p1)
                if hasattr(self.plugin, 'resonance'): self.plugin.resonance = self.m_p2
        except Exception: pass
    def process(self, audio_bus, sample_rate):
        with self.lock:
            self._apply_params()
            if self.fx_type in["Distortion", "Resonant LPF"]:
                return audio_bus * (1.0 - (self.m_p2 if self.fx_type == "Distortion" else 1.0)) + self.plugin.process(audio_bus, sample_rate, reset=False) * (self.m_p2 if self.fx_type == "Distortion" else 1.0)
            elif self.fx_type == "Compressor": return self.plugin.process(audio_bus, sample_rate, reset=False) * (10.0 ** ((self.m_p3 * 24.0) / 20.0))
            else: return self.plugin.process(audio_bus, sample_rate, reset=False)

class DrumbExtractor:
    def __init__(self): 
        self.last_filepath = None
        self.used_starts = set()

    def _generate_random_samples(self, y, sr, num, length):
        """Helper to generate entirely randomized pitch-shifted arrays safely"""
        samples =[]
        snip_len = int(SAMPLE_RATE * length)
        max_start = max(0, len(y) - snip_len)
        for _ in range(num):
            start = random.randint(0, max_start)
            snip = y[start:start + snip_len]
            if random.random() < 0.75: 
                snip = librosa.effects.pitch_shift(snip, sr=sr, n_steps=random.uniform(-24, 24))
            samples.append(snip)
        return samples

    def extract(self, filepath, old_way=False, is_rescan=False):
        if filepath != self.last_filepath or not is_rescan: 
            self.used_starts.clear()
            self.last_filepath = filepath
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        return self._extract_old(y, sr) if old_way else self._extract_new(y, sr)

    def rescan_single_channel(self, filepath, channel_name):
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        if channel_name == "Random": return self._generate_random_samples(y, sr, 15, 0.3)
        
        candidates, new_samples, sl = [],[], int(SAMPLE_RATE * 0.3)
        for start in librosa.frames_to_samples(librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)):
            if start in self.used_starts: continue
            snippet = y[start:start+sl]
            if len(snippet) == 0 or np.mean(librosa.feature.rms(y=snippet)) < 0.005: continue
            snippet = snippet / np.max(np.abs(snippet)) if np.max(np.abs(snippet)) > 0 else snippet
            candidates.append({'audio': snippet, 'cent': np.mean(librosa.feature.spectral_centroid(y=snippet, sr=sr)), 'zcr': np.mean(librosa.feature.zero_crossing_rate(y=snippet)), 'start': start})
            
        def add_if(f):
            nonlocal candidates
            valid =[c for c in candidates if f(c)]; random.shuffle(valid)
            for v in valid:
                if len(new_samples) >= 15: break
                self.used_starts.add(v['start']); new_samples.append(v['audio']); candidates = [c for c in candidates if c['start'] != v['start']]
                
        strict = {"Kick": lambda c: c['cent'] < 1200, "Snare": lambda c: c['zcr'] > 0.12 and 1500 < c['cent'] < 4000, "Closed Hat": lambda c: c['cent'] > 5500, "Open Hat": lambda c: c['cent'] > 4500, "Hi Tom": lambda c: 1000 < c['cent'] < 2000, "Low Tom": lambda c: 500 < c['cent'] < 1200, "Crash": lambda c: c['cent'] > 3000}
        relax = {"Kick": lambda c: c['cent'] < 2000, "Snare": lambda c: c['zcr'] > 0.07, "Closed Hat": lambda c: c['cent'] > 3500, "Open Hat": lambda c: c['cent'] > 3000, "Hi Tom": lambda c: 800 < c['cent'] < 2500, "Low Tom": lambda c: 300 < c['cent'] < 1500, "Crash": lambda c: c['cent'] > 2000}
        if channel_name in strict: add_if(strict[channel_name])
        if channel_name in relax: add_if(relax[channel_name])
        while len(new_samples) < 15:
            if candidates:
                c = random.choice(candidates); self.used_starts.add(c['start']); new_samples.append(c['audio']); candidates =[cand for cand in candidates if cand['start'] != c['start']]
            else: new_samples.append(np.zeros(sl, dtype=np.float32))
        return new_samples

    def _extract_new(self, y, sr):
        candidates, extracted, sl =[], {c:[] for c in CHANNELS}, int(SAMPLE_RATE * 0.3) 
        for start in librosa.frames_to_samples(librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)):
            if start in self.used_starts: continue 
            snippet = y[start:start+sl]
            if len(snippet) == 0 or np.mean(librosa.feature.rms(y=snippet)) < 0.005: continue
            snippet = snippet / np.max(np.abs(snippet)) if np.max(np.abs(snippet)) > 0 else snippet
            candidates.append({'audio': snippet, 'cent': np.mean(librosa.feature.spectral_centroid(y=snippet, sr=sr)), 'zcr': np.mean(librosa.feature.zero_crossing_rate(y=snippet)), 'start': start})
            
        def run_pass(ch, filt):
            nonlocal candidates
            if (needs := 15 - len(extracted[ch])) <= 0: return
            valid =[c for c in candidates if filt(c)]; random.shuffle(valid)
            for v in valid[:needs]: self.used_starts.add(v['start']); extracted[ch].append(v['audio']); candidates =[c for c in candidates if c['start'] != v['start']]
            
        for m in[{"Kick": lambda c: c['cent'] < 1200, "Snare": lambda c: c['zcr'] > 0.12 and 1500 < c['cent'] < 4000, "Closed Hat": lambda c: c['cent'] > 5500, "Open Hat": lambda c: c['cent'] > 4500, "Hi Tom": lambda c: 1000 < c['cent'] < 2000, "Low Tom": lambda c: 500 < c['cent'] < 1200, "Crash": lambda c: c['cent'] > 3000},
                  {"Kick": lambda c: c['cent'] < 2000, "Snare": lambda c: c['zcr'] > 0.07, "Closed Hat": lambda c: c['cent'] > 3500, "Open Hat": lambda c: c['cent'] > 3000, "Hi Tom": lambda c: 800 < c['cent'] < 2500, "Low Tom": lambda c: 300 < c['cent'] < 1500, "Crash": lambda c: c['cent'] > 2000}]:
            for ch, rule in m.items(): run_pass(ch, rule)
            
        for ch in extracted:
            while len(extracted[ch]) < 15:
                if candidates: c = random.choice(candidates); self.used_starts.add(c['start']); extracted[ch].append(c['audio']); candidates =[cand for cand in candidates if cand['start'] != c['start']]
                else: extracted[ch].append(np.zeros(sl, dtype=np.float32))
                
        extracted["Random"] = self._generate_random_samples(y, sr, 15, 0.3)
        return extracted

    def _extract_old(self, y, sr):
        drumbs, onset_samples =[], librosa.frames_to_samples(librosa.onset.onset_detect(y=y, sr=sr, backtrack=True))
        for i, start in enumerate(onset_samples):
            if start in self.used_starts: continue
            snippet = y[start:min(onset_samples[i+1] if i+1 < len(onset_samples) else len(y), start + int(SAMPLE_RATE * 0.5))]
            if len(snippet) < 1000: continue 
            snippet = snippet / np.max(np.abs(snippet)) if np.max(np.abs(snippet)) > 0 else snippet
            drumbs.append({'audio': snippet, 'centroid': np.mean(librosa.feature.spectral_centroid(y=snippet, sr=sr)), 'zcr': np.mean(librosa.feature.zero_crossing_rate(y=snippet)), 'duration': len(snippet)/sr, 'start': start})
        if not drumbs: raise ValueError("No unused transients found.")
        
        def mark_used(items):
            for i in items: self.used_starts.add(i['start'])
            return[i['audio'] for i in items]
            
        s_cent = sorted(drumbs, key=lambda x: x['centroid']); s_zcr = sorted(drumbs, key=lambda x: x['zcr'], reverse=True); s_dur = sorted(drumbs, key=lambda x: x['duration'], reverse=True)
        extracted = {
            "Kick": mark_used(s_cent[:15]), "Snare": mark_used(s_zcr[:15]),
            "Closed Hat": mark_used(sorted(s_cent[-30:], key=lambda x: x['duration'])[:15]), "Open Hat": mark_used(sorted(s_cent[-30:], key=lambda x: x['duration'], reverse=True)[:15]),
            "Hi Tom": mark_used(s_cent[15:30]), "Low Tom": mark_used(s_cent[30:45]), "Crash": mark_used(s_dur[:15]),
            "Random": self._generate_random_samples(y, sr, 15, 0.1)
        }
        for k in extracted:
            while len(extracted[k]) < 15: extracted[k].append(extracted[k][0] if extracted[k] else np.zeros(100))
        return extracted

class DrumChannel:
    def __init__(self, name, engine_ref):
        self.name, self.engine, self.samples = name, engine_ref,[np.zeros(1024, dtype=np.float32)] * 15
        self.raw_samples =[np.zeros(1024, dtype=np.float32)] * 15
        self.current_sample_idx = 0; self.mute, self.solo, self.solo_locked, self.locked, self.trigger_flag = False, False, False, False, False
        self.pattern_settings =[]
        for _ in range(4): self.pattern_settings.append({'vol':0.8, 'pan':0.0, 'pitch':0, 's_start':0, 's_end':200, 'fx1':0.0, 'fx2':0.0, 'fx3':0.0, 'euc':0})
        self.vol, self.pan, self.pitch, self.sample_start, self.sample_end, self.fx1, self.fx2, self.fx3, self.euclid_k = 0.8, 0.0, 0, 0, 200, 0.0, 0.0, 0.0, 0
        self.m_vol, self.m_pan, self.m_pitch, self.m_sample_start, self.m_sample_end, self.m_fx1, self.m_fx2, self.m_fx3, self.m_euc = 0.8, 0.0, 0, 0, 200, 0.0, 0.0, 0.0, 0
        self.m_mute, self.m_solo = False, False
        self.sequence = [[False]*32 for _ in range(4)]; self.m_sequence = [[False]*32 for _ in range(4)]
        
    def save_pattern_state(self, pat_idx):
        self.pattern_settings[pat_idx] = {'vol': self.vol, 'pan': self.pan, 'pitch': self.pitch, 's_start': self.sample_start, 's_end': self.sample_end, 'fx1': self.fx1, 'fx2': self.fx2, 'fx3': self.fx3, 'euc': self.euclid_k}
        
    def load_pattern_state(self, pat_idx):
        s = self.pattern_settings[pat_idx]
        self.vol, self.pan, self.pitch, self.sample_start, self.sample_end = s['vol'], s['pan'], s['pitch'], s['s_start'], s['s_end']
        self.fx1, self.fx2, self.fx3, self.euclid_k = s['fx1'], s['fx2'], s['fx3'], s['euc']

    def get_state(self):
        return {'idx': self.current_sample_idx, 'mute': self.mute, 'solo_l': self.solo_locked, 'locked': self.locked, 'ps': copy.deepcopy(self.pattern_settings), 'seq': [r[:] for r in self.sequence]}
        
    def apply_state(self, s, pat):
        self.current_sample_idx, self.mute, self.solo_locked, self.locked, self.pattern_settings = s['idx'], s['mute'], s['solo_l'], s.get('locked', False), copy.deepcopy(s['ps'])
        self.solo, self.sequence = self.solo_locked, [r[:] for r in s['seq']]
        self.load_pattern_state(pat)

    def step_sample(self, forward=True):
        self.current_sample_idx = (self.current_sample_idx + (1 if forward else -1)) % 15; self.render()
        
    def render(self):
        raw = self.samples[self.current_sample_idx].copy()
        start_samp = max(0, min(len(raw)-1, int((self.sample_start / 1000.0) * SAMPLE_RATE)))
        end_samp = max(start_samp+1, min(len(raw), int((self.sample_end / 1000.0) * SAMPLE_RATE)))
        raw = raw[start_samp:end_samp]
        if (tp := self.pitch + self.engine.global_pitch) != 0: raw = librosa.effects.pitch_shift(raw, sr=SAMPLE_RATE, n_steps=tp)
        if (fade := min(int(0.01 * SAMPLE_RATE), len(raw))) > 0: raw[-fade:] *= np.linspace(1, 0, fade)
        self.audio_buffer = np.array(raw, dtype=np.float32)

class AudioEngine:
    def __init__(self):
        self.channels =[DrumChannel(name, self) for name in CHANNELS]
        self.fx_buses =[GlobalFXBus("Reverb"), GlobalFXBus("Resonant LPF"), GlobalFXBus("Compressor")]
        self.lfos =[LFO("#00FFFF"), LFO("#FF00FF"), LFO("#FFFF00"), LFO("#00FF00")]
        self.active_voices, self.record_buffer, self.is_playing, self.is_recording =[],[], False, False
        self.lfo_trigger_queue =[]
        self.bpm, self.steps, self.current_pattern, self.swing, self.burst_multiplier = 120, 16, 0, 0.0, 1.0
        self.global_vol, self.global_pitch, self.master_fx1, self.master_fx2, self.master_fx3 = 0.8, 0, 0.0, 0.0, 0.0
        self.m_global_vol, self.m_global_pitch, self.m_swing, self.m_master_fx1, self.m_master_fx2, self.m_master_fx3 = 0.8, 0, 0.0, 0.0, 0.0, 0.0
        self.samples_until_next_step, self.step_counter, self.master_samples_until_next_step, self.master_step_counter, self.was_bursting = 0.0, 0, 0.0, 0, False
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=2, callback=self.audio_callback, dtype='float32')
        self.stream.start()
        
    def get_state(self):
        return {'bpm': self.bpm, 'steps': self.steps, 'pat': self.current_pattern, 'swing': self.swing, 'gvol': self.global_vol, 'gpitch': self.global_pitch, 'm_fx1': self.master_fx1, 'm_fx2': self.master_fx2, 'm_fx3': self.master_fx3, 'ch':[c.get_state() for c in self.channels], 'fx':[{'type': f.fx_type, 'p1': f.p1, 'p2': f.p2, 'p3': f.p3} for f in self.fx_buses]}
        
    def apply_state(self, s):
        self.bpm, self.steps, self.current_pattern, self.swing, self.global_vol, self.global_pitch = s['bpm'], s['steps'], s['pat'], s['swing'], s['gvol'], s['gpitch']
        self.master_fx1, self.master_fx2, self.master_fx3 = s.get('m_fx1', 0.0), s.get('m_fx2', 0.0), s.get('m_fx3', 0.0)
        for i, c in enumerate(self.channels): c.apply_state(s['ch'][i], self.current_pattern)
        for i, f in enumerate(self.fx_buses):
            fs = s['fx'][i]
            if f.fx_type != fs['type']: f.set_type(fs['type'])
            f.set_param(1, fs['p1']); f.set_param(2, fs['p2']); f.set_param(3, fs['p3'])

    def apply_lfo_target(self, target, val, depth):
        parts, offset = target.split(":"), val * depth
        if parts[0] == "ch":
            ch, param = self.channels[int(parts[1])], parts[2]
            if param in['cyc', 'trigger']: return
            if param in['mute', 'solo']: setattr(ch, "m_"+param, (float(getattr(ch, param)) + offset) > 0.5)
            else:
                limits = {'vol':(0,1), 'pan':(-1,1), 'pitch':(-24,24), 'fx1':(0,1), 'fx2':(0,1), 'fx3':(0,1), 'sample_start':(0,300), 'sample_end':(0,300), 'euc':(0, self.steps)}
                b, span = getattr(ch, param), limits[param][1] - limits[param][0]
                setattr(ch, "m_"+param, max(limits[param][0], min(limits[param][1], b + offset * span)))
        elif parts[0] == "gl":
            param = parts[1]; limits = {'global_vol':(0,1), 'global_pitch':(-24,24), 'swing':(0,0.5), 'master_fx1':(0,1), 'master_fx2':(0,1), 'master_fx3':(0,1)}
            b, span = getattr(self, param), limits[param][1] - limits[param][0]
            setattr(self, "m_"+param, max(limits[param][0], min(limits[param][1], b + offset * span)))
        elif parts[0] == "fx":
            fx_b, p_idx = self.fx_buses[int(parts[1])], int(parts[2][1]) - 1
            b, p_min, p_max = getattr(fx_b, parts[2]), FX_DEFS[fx_b.fx_type][p_idx][1], FX_DEFS[fx_b.fx_type][p_idx][2]
            setattr(fx_b, "m_"+parts[2], max(p_min, min(p_max, b + offset * (p_max - p_min))))
        elif parts[0] == "grid":
            ch, step = self.channels[int(parts[1])], int(parts[2])
            ch.m_sequence[self.current_pattern][step] = (float(ch.sequence[self.current_pattern][step]) + offset) > 0.5

    def trigger_channel(self, ch):
        ch.trigger_flag = True; pan = ch.m_pan
        lg, rg = math.cos((pan + 1) * math.pi / 4) * ch.m_vol, math.sin((pan + 1) * math.pi / 4) * ch.m_vol
        raw = ch.samples[ch.current_sample_idx]
        st, en = max(0, min(len(raw)-1, int((ch.m_sample_start / 1000.0) * SAMPLE_RATE))), max(max(0, min(len(raw)-1, int((ch.m_sample_start / 1000.0) * SAMPLE_RATE)))+1, min(len(raw), int((ch.m_sample_end / 1000.0) * SAMPLE_RATE)))
        buf = raw[st:en]
        if (fade := min(int(0.01 * SAMPLE_RATE), len(buf))) > 0: buf = buf.copy(); buf[-fade:] *= np.linspace(1, 0, fade)
        self.active_voices.append({'ch': ch, 'buffer': buf, 'pos': 0.0, 'len': len(buf), 'lg': lg, 'rg': rg, 'fx1': ch.m_fx1, 'fx2': ch.m_fx2, 'fx3': ch.m_fx3})

    def audio_callback(self, outdata, frames, time_info, status):
        for ch in self.channels:
            ch.m_vol, ch.m_pan, ch.m_pitch, ch.m_sample_start, ch.m_sample_end = ch.vol, ch.pan, ch.pitch, ch.sample_start, ch.sample_end
            ch.m_fx1, ch.m_fx2, ch.m_fx3, ch.m_euc, ch.m_mute, ch.m_solo = ch.fx1, ch.fx2, ch.fx3, ch.euclid_k, ch.mute, ch.solo
            for p in range(4):
                for s in range(self.steps): ch.m_sequence[p][s] = ch.sequence[p][s]
        self.m_global_vol, self.m_global_pitch, self.m_swing = self.global_vol, self.global_pitch, self.swing
        self.m_master_fx1, self.m_master_fx2, self.m_master_fx3 = self.master_fx1, self.master_fx2, self.master_fx3
        for fx in self.fx_buses: fx.m_p1, fx.m_p2, fx.m_p3 = fx.p1, fx.p2, fx.p3

        time_delta = frames / SAMPLE_RATE
        for lfo in self.lfos:
            lfo.step(time_delta, self.bpm, self.lfo_trigger_queue)
            if lfo.target_id and lfo.depth > 0 and not (lfo.target_id.startswith("gl_rand:") or lfo.target_id.endswith(":cyc") or lfo.target_id.endswith(":trigger")): 
                self.apply_lfo_target(lfo.target_id, lfo.val, lfo.depth)

        for ch in self.channels:
            if ch.m_euc != ch.euclid_k and ch.m_euc > 0:
                k, st = int(ch.m_euc), self.steps
                for i in range(st): ch.m_sequence[self.current_pattern][i] = ((i * k) % st) < k

        out, fx1_bus, fx2_bus, fx3_bus = np.zeros((frames, 2), dtype=np.float32), np.zeros((frames, 2), dtype=np.float32), np.zeros((frames, 2), dtype=np.float32), np.zeros((frames, 2), dtype=np.float32)
        any_soloed = any(ch.m_solo for ch in self.channels)
        
        if self.is_playing:
            if self.burst_multiplier == 1.0 and self.was_bursting:
                self.step_counter, self.samples_until_next_step, self.was_bursting = self.master_step_counter, self.master_samples_until_next_step, False
            elif self.burst_multiplier != 1.0: self.was_bursting = True

            frame_idx = 0
            while frame_idx < frames:
                if self.samples_until_next_step <= 0:
                    step_idx = self.step_counter % self.steps
                    for ch in self.channels:
                        if ch.m_sequence[self.current_pattern][step_idx] and not ch.m_mute:
                            if not any_soloed or ch.m_solo: self.trigger_channel(ch)
                    self.step_counter += 1
                    base_sps = (60.0 / max(1.0, float(self.bpm * self.burst_multiplier))) * (SAMPLE_RATE / 4.0)
                    sw = base_sps * float(self.m_swing)
                    self.samples_until_next_step += (base_sps + sw) if self.step_counter % 2 == 1 else (base_sps - sw)
                
                if self.master_samples_until_next_step <= 0:
                    self.master_step_counter += 1
                    base_sps_m = (60.0 / max(1.0, float(self.bpm))) * (SAMPLE_RATE / 4.0)
                    sw_m = base_sps_m * float(self.m_swing)
                    self.master_samples_until_next_step += (base_sps_m + sw_m) if self.master_step_counter % 2 == 1 else (base_sps_m - sw_m)

                advance = min(frames - frame_idx, max(1, int(math.ceil(min(self.samples_until_next_step, self.master_samples_until_next_step)))))
                self.samples_until_next_step -= advance; self.master_samples_until_next_step -= advance; frame_idx += advance

        active_next =[]
        for v in self.active_voices:
            ch = v['ch']
            tot_pitch = ch.m_pitch + self.m_global_pitch
            ratio = 2.0 ** (tot_pitch / 12.0)
            end_pos = v['pos'] + frames * ratio
            
            if end_pos > v['len']:
                write_frames = int(math.ceil((v['len'] - v['pos']) / ratio)); end_pos = v['len']
            else: write_frames = frames

            if write_frames > 0:
                indices = np.linspace(v['pos'], end_pos, write_frames, endpoint=False)
                np.clip(indices, 0, v['len']-1, out=indices)
                audio_slice = np.interp(indices, np.arange(v['len']), v['buffer'])

                sl_lg, sl_rg = audio_slice * ch.m_vol * math.cos((ch.m_pan + 1) * math.pi / 4), audio_slice * ch.m_vol * math.sin((ch.m_pan + 1) * math.pi / 4)
                out[:write_frames, 0] += sl_lg; out[:write_frames, 1] += sl_rg
                if ch.m_fx1 > 0: fx1_bus[:write_frames, 0] += sl_lg * ch.m_fx1; fx1_bus[:write_frames, 1] += sl_rg * ch.m_fx1
                if ch.m_fx2 > 0: fx2_bus[:write_frames, 0] += sl_lg * ch.m_fx2; fx2_bus[:write_frames, 1] += sl_rg * ch.m_fx2
                if ch.m_fx3 > 0: fx3_bus[:write_frames, 0] += sl_lg * ch.m_fx3; fx3_bus[:write_frames, 1] += sl_rg * ch.m_fx3
            
            v['pos'] = end_pos
            if v['pos'] < v['len']: active_next.append(v)
                
        self.active_voices = active_next
        sig1 = (out * (1.0 - self.m_master_fx1)) + self.fx_buses[0].process((out * self.m_master_fx1) + fx1_bus, SAMPLE_RATE)
        sig2 = (sig1 * (1.0 - self.m_master_fx2)) + self.fx_buses[1].process((sig1 * self.m_master_fx2) + fx2_bus, SAMPLE_RATE)
        sig3 = (sig2 * (1.0 - self.m_master_fx3)) + self.fx_buses[2].process((sig2 * self.m_master_fx3) + fx3_bus, SAMPLE_RATE)
        
        outdata[:] = sig3 * self.m_global_vol
        if self.is_recording: self.record_buffer.append(outdata.copy())


class DrumMachineApp(ctk.CTk):
    def __init__(self, engine):
        super().__init__()
        self.engine, self.extractor, self.current_audio_file, self.first_load_done = engine, DrumbExtractor(), None, False
        self.title("Drumber v8.3"); self.geometry("1450x850"); ctk.set_appearance_mode("Dark"); self.configure(fg_color="#0A0A0A") 
        self.ch_ui_refs, self.fx_ui_refs, self.lfo_ui_refs, self.g_ui_refs, self.g_lbl_refs = {},[],[], {}, {}
        self.widget_to_step, self.grid_buttons, self.widget_to_lfo_target = {}, {}, {}
        self.target_updater_map = {}
        
        self.undo_stack = deque(maxlen=24)
        self.rand_history = {k: deque(maxlen=8) for k in['Cyc', 'Euc', 'Sam', 'Pan', 'Pit', 'ALL']}
        self.mute_history, self.solo_history = deque(maxlen=8), deque(maxlen=8)
        self.use_old_extraction = ctk.BooleanVar(value=False)
        self.ctrl_pressed = False
        
        self.bind_all("<B1-Motion>", lambda e: self.on_paint(e, True)); self.bind_all("<B3-Motion>", lambda e: self.on_paint(e, False))
        self.bind_all("<ButtonPress-1>", self.on_global_click_save_state, add="+"); self.bind_all("<ButtonPress-3>", lambda e: self.on_paint(e, False), add="+")
        self.bind_all("<KeyPress-Control_L>", lambda e: setattr(self, 'ctrl_pressed', True)); self.bind_all("<KeyRelease-Control_L>", lambda e: setattr(self, 'ctrl_pressed', False))
        self.bind_all("<KeyPress-Control_R>", lambda e: setattr(self, 'ctrl_pressed', True)); self.bind_all("<KeyRelease-Control_R>", lambda e: setattr(self, 'ctrl_pressed', False))
        self.bind_all("<Button-2>", self.show_lfo_menu)

        self.build_ui(); self.save_state(); self.rand_pan(); self.gui_update_loop()
        
    def gui_update_loop(self):
        while self.engine.lfo_trigger_queue:
            t_id = self.engine.lfo_trigger_queue.pop(0)
            if t_id.startswith("gl_rand:"): self.do_global_rand(t_id.split(":")[1])
            elif t_id.startswith("ch:") and t_id.endswith(":cyc"): self.cycle_drumb(self.engine.channels[int(t_id.split(":")[1])])
            elif t_id.startswith("ch:") and t_id.endswith(":trigger"): self.engine.trigger_channel(self.engine.channels[int(t_id.split(":")[1])])
            
        for i, ch in enumerate(self.engine.channels):
            if ch.trigger_flag:
                ch.trigger_flag = False; led = self.ch_ui_refs[ch.name]['led']; led.configure(fg_color=ROW_COLORS[i])
                self.after(100, lambda l=led: l.configure(fg_color="#222222"))
                
            m_col = "#CC0000" if ch.m_mute else "#444444"
            btn_mute = self.ch_ui_refs[ch.name]['btn_mute']
            if btn_mute.cget("fg_color") != m_col: btn_mute.configure(fg_color=m_col)
                
            s_col = "#CCCC00" if ch.solo_locked else ("#FF9900" if ch.m_solo else "#444444")
            btn_solo = self.ch_ui_refs[ch.name]['btn_solo']
            if btn_solo.cget("fg_color") != s_col: btn_solo.configure(fg_color=s_col)
            
        self.refresh_grid_ui()

        for i, lfo in enumerate(self.engine.lfos):
            bright = int(((lfo.val + 1.0) / 2.0) * 255)
            self.lfo_ui_refs[i]['led'].configure(fg_color=f"#{bright:02x}{bright:02x}{bright:02x}")
            if lfo.target_id and not (lfo.target_id.startswith("gl_rand:") or lfo.target_id.endswith(":cyc") or lfo.target_id.endswith(":trigger")) and lfo.target_id in self.target_updater_map:
                try: self.target_updater_map[lfo.target_id]()
                except Exception: pass
                
        self.after(30, self.gui_update_loop)
        
    def save_state(self): self.undo_stack.append(self.engine.get_state())
    def undo(self):
        if len(self.undo_stack) > 1:
            self.undo_stack.pop(); self.engine.apply_state(self.undo_stack[-1]); self.sync_ui_to_engine(); self.lbl_status.configure(text="Global undo successful.")

    def _save_rand_undo(self, key):
        attrs =['current_sample_idx'] if key=='Cyc' else['euclid_k', 'sequence'] if key=='Euc' else['sample_start', 'sample_end'] if key=='Sam' else ['pan'] if key=='Pan' else ['pitch'] if key=='Pit' else['current_sample_idx', 'euclid_k', 'sequence', 'sample_start', 'sample_end', 'pan', 'pitch']
        snap = {}
        for ch in self.engine.channels:
            if not ch.locked: snap[ch.name] = {a: [r[:] for r in ch.sequence] if a == 'sequence' else getattr(ch, a) for a in attrs}
        self.rand_history[key].append(snap)

    def _apply_rand_undo(self, key):
        if not self.rand_history[key]: return
        snap = self.rand_history[key].pop()
        for ch in self.engine.channels:
            if ch.name in snap:
                for a, v in snap[ch.name].items():
                    if a == 'sequence': ch.sequence =[r[:] for r in v]
                    else: setattr(ch, a, v)
        self.sync_ui_to_engine(); self.lbl_status.configure(text=f"Reverted specific {key} randomization.")

    def on_global_click_save_state(self, e):
        try:
            if (w := self.winfo_containing(e.x_root, e.y_root)) and (str(w) in self.widget_to_step or "slider" in w.winfo_class().lower() or "canvas" in w.winfo_class().lower()): self.save_state()
        except Exception: pass

    def show_lfo_menu(self, e):
        w = self.winfo_containing(e.x_root, e.y_root)
        if not w: return
        t_id, t_name = None, None
        curr = w
        while curr:
            if str(curr) in self.widget_to_lfo_target:
                t_id, t_name = self.widget_to_lfo_target[str(curr)]; break
            curr = curr.master
        if t_id:
            menu = tk.Menu(self, tearoff=0, bg="#222", fg="white", font=APP_FONT)
            menu.add_command(label=f"Assign '{t_name}' to LFO 1", command=lambda: self.assign_lfo(0, t_id, t_name))
            menu.add_command(label=f"Assign '{t_name}' to LFO 2", command=lambda: self.assign_lfo(1, t_id, t_name))
            menu.add_command(label=f"Assign '{t_name}' to LFO 3", command=lambda: self.assign_lfo(2, t_id, t_name))
            menu.add_command(label=f"Assign '{t_name}' to LFO 4", command=lambda: self.assign_lfo(3, t_id, t_name))
            menu.tk_popup(e.x_root, e.y_root)

    def assign_lfo(self, idx, t_id, t_name):
        self.engine.lfos[idx].target_id = t_id; self.engine.lfos[idx].target_name = t_name
        self.lfo_ui_refs[idx]['lbl_target'].configure(text=t_name)
        
    def do_global_rand(self, key):
        self.save_state(); self._save_rand_undo(key)
        if key == 'Cyc': self.rand_cycl()
        elif key == 'Euc': self.rand_euc()
        elif key == 'Sam': self.rand_sam()
        elif key == 'Pan': self.rand_pan()
        elif key == 'Pit': self.rand_pitch()
        elif key == 'ALL': self.rand_all()

    def build_ui(self):
        top_frame = ctk.CTkFrame(self, fg_color="#141414"); top_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(top_frame, text="Load & Scan", font=APP_FONT_BOLD, width=100, command=self.load_file).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="RESCAN", font=APP_FONT_BOLD, width=80, fg_color="#B9770E", command=self.rescan_file).pack(side="left", padx=5)
        self.lbl_status = ctk.CTkLabel(top_frame, text="Ready. Load an audio file.", font=APP_FONT); self.lbl_status.pack(side="left", padx=10)
        ctk.CTkButton(top_frame, text="⚙", width=30, font=APP_FONT, fg_color="transparent", border_width=1, command=self.open_options).pack(side="right", padx=5)
        ctk.CTkButton(top_frame, text="PLAY / STOP", font=APP_FONT_BOLD, command=self.toggle_play, fg_color="#228B22").pack(side="right", padx=5)
        self.btn_rec = ctk.CTkButton(top_frame, text="RECORD", font=APP_FONT_BOLD, command=self.toggle_record, fg_color="transparent", border_width=1); self.btn_rec.pack(side="right", padx=10)

        global_frame = ctk.CTkFrame(self, fg_color="#141414"); global_frame.pack(fill="x", padx=10, pady=5)
        self.add_global_knob(global_frame, "Vol", 0, 1, 0.8, lambda v: setattr(self.engine, 'global_vol', v), "gl:global_vol")
        self.add_global_knob(global_frame, "FX1 Send", 0, 1, 0.0, lambda v: setattr(self.engine, 'master_fx1', v), "gl:master_fx1")
        self.add_global_knob(global_frame, "FX2 Send", 0, 1, 0.0, lambda v: setattr(self.engine, 'master_fx2', v), "gl:master_fx2")
        self.add_global_knob(global_frame, "FX3 Send", 0, 1, 0.0, lambda v: setattr(self.engine, 'master_fx3', v), "gl:master_fx3")
        self.add_global_knob(global_frame, "Pitch", -24, 24, 0, lambda v: setattr(self.engine, 'global_pitch', v), "gl:global_pitch", is_int=True)
        
        self.bpm_widget = DraggableBPM(global_frame, self, default=120); self.bpm_widget.pack(side="left", padx=5)
        
        burst_f = ctk.CTkFrame(global_frame, fg_color="transparent"); burst_f.pack(side="left", padx=2)
        burst_k = CTkKnob(burst_f, width=40, height=40, from_=0, to=7); burst_k.set(5); burst_k.pack()
        b_ratios =[0.5, 0.66, 1.0, 1.33, 1.5, 2.0, 4.0, 8.0]
        lbl_b = ctk.CTkLabel(burst_f, text="x2.0", font=("Trebuchet MS", 10)); lbl_b.pack()
        burst_k.command = lambda v: lbl_b.configure(text=f"x{b_ratios[int(round(v))]}")
        btn_burst = ctk.CTkButton(burst_f, text="BURST", width=40, font=APP_FONT_BOLD, fg_color="#E67E22", hover_color="#D35400")
        btn_burst.pack(); btn_burst.bind("<ButtonPress-1>", lambda e: setattr(self.engine, 'burst_multiplier', b_ratios[int(round(burst_k.get()))]))
        btn_burst.bind("<ButtonRelease-1>", lambda e: setattr(self.engine, 'burst_multiplier', 1.0))
        
        self.add_global_knob(global_frame, "Swing", 0, 0.5, 0.0, lambda v: setattr(self.engine, 'swing', v), "gl:swing")
        
        pat_frame = ctk.CTkFrame(global_frame, fg_color="transparent"); pat_frame.pack(side="left", padx=5)
        ctk.CTkLabel(pat_frame, text="Pattern", font=APP_FONT).pack(side="left", padx=5)
        self.pat_var = ctk.StringVar(value="0"); ctk.CTkSegmentedButton(pat_frame, values=["0", "1", "2", "3"], font=APP_FONT, variable=self.pat_var, command=self.change_pattern).pack(side="left")
                               
        step_frame = ctk.CTkFrame(global_frame, fg_color="transparent"); step_frame.pack(side="left", padx=5)
        ctk.CTkLabel(step_frame, text="Steps", font=APP_FONT).pack(side="left", padx=5)
        self.step_var = ctk.StringVar(value="16"); ctk.CTkSegmentedButton(step_frame, values=["8", "16", "32"], font=APP_FONT, variable=self.step_var, command=self.change_steps).pack(side="left")

        rand_frame = ctk.CTkFrame(global_frame, fg_color="transparent"); rand_frame.pack(side="left", padx=10)
        def mk_rnd(txt, key, col=None, dbl=None):
            b = ctk.CTkButton(rand_frame, text=txt, width=30, font=APP_FONT, command=lambda: self.do_global_rand(key))
            if col: b.configure(fg_color=col)
            b.pack(side="left", padx=2); b.bind("<Button-3>", lambda e: self._apply_rand_undo(key))
            if dbl: b.bind("<Double-Button-1>", lambda e: dbl())
            self.widget_to_lfo_target[str(b)] = (f"gl_rand:{key}", f"Rand {txt}")
            
        ctk.CTkLabel(rand_frame, text="Rand:", font=APP_FONT).pack(side="left", padx=5)
        mk_rnd("ALL", 'ALL', "#A93226"); mk_rnd("Cyc", 'Cyc'); mk_rnd("Euc", 'Euc'); mk_rnd("Sam", 'Sam', dbl=self.reset_rand_sam); mk_rnd("Pan", 'Pan'); mk_rnd("Pit", 'Pit')
        
        act_f = ctk.CTkFrame(global_frame, fg_color="transparent"); act_f.pack(side="right", padx=5)
        ctk.CTkButton(act_f, text="RESET ALL", width=60, font=APP_FONT_BOLD, command=self.reset_all_params, fg_color="#6B1D1D").pack(side="right", padx=2)
        ctk.CTkButton(act_f, text="UNDO", width=50, font=APP_FONT_BOLD, command=self.undo, fg_color="#1E8449").pack(side="right", padx=2)
        btn_uns = ctk.CTkButton(act_f, text="UNSOLO", width=50, font=APP_FONT_BOLD, command=self.unsolo_all, fg_color="#9C640C"); btn_uns.bind("<Button-3>", self.undo_unsolo_all); btn_uns.pack(side="right", padx=2)
        btn_unm = ctk.CTkButton(act_f, text="UNMUTE", width=50, font=APP_FONT_BOLD, command=self.unmute_all, fg_color="#922B21"); btn_unm.bind("<Button-3>", self.undo_unmute_all); btn_unm.pack(side="right", padx=2)

        main_paned = ctk.CTkFrame(self, fg_color="transparent"); main_paned.pack(fill="both", expand=True, padx=10, pady=(5,0))
        
        top_band = ctk.CTkFrame(main_paned, fg_color="transparent"); top_band.pack(fill="both", expand=True)
        mixer_container = ctk.CTkFrame(top_band, fg_color="transparent"); mixer_container.pack(side="left", fill="both", expand=True) 
        mixer_frame = ctk.CTkFrame(mixer_container, fg_color="#111111"); mixer_frame.pack(pady=2, fill="both", expand=True)
        
        self.fx_frame = ctk.CTkScrollableFrame(top_band, width=300, fg_color="#141414"); self.fx_frame.pack(side="right", fill="y", padx=(5,0))

        for i, ch in enumerate(self.engine.channels):
            self.ch_ui_refs[ch.name] = {}
            col = ctk.CTkFrame(mixer_frame, width=130, fg_color="#1A1A1A"); col.pack(side="left", fill="y", expand=True, padx=2, pady=1) 
            top_cf = ctk.CTkFrame(col, fg_color="transparent"); top_cf.pack(pady=6)
            led = ctk.CTkFrame(top_cf, width=8, height=8, corner_radius=4, fg_color="#222222"); led.pack(side="left", padx=2); self.ch_ui_refs[ch.name]['led'] = led
            b_tit = ctk.CTkButton(top_cf, text=ch.name, width=60, height=22, corner_radius=11, font=APP_FONT_BOLD, fg_color=ROW_COLORS[i], hover_color="#888888", text_color="#000000", command=lambda c=ch: self.engine.trigger_channel(c)); b_tit.pack(side="left", padx=2)
            self.widget_to_lfo_target[str(b_tit)] = (f"ch:{i}:trigger", f"{ch.name} Trigger")
            btn_lock = ctk.CTkButton(top_cf, text="🔓", width=22, height=22, font=("Arial", 12), fg_color="#333333", hover_color="#555555")
            btn_lock.configure(command=lambda c=ch, b=btn_lock: self.toggle_lock(c, b)); btn_lock.pack(side="left", padx=2); self.ch_ui_refs[ch.name]['btn_lock'] = btn_lock
            
            btn_frame = ctk.CTkFrame(col, fg_color="transparent"); btn_frame.pack(fill="x", pady=1)
            btn_mute = ctk.CTkButton(btn_frame, text="M", width=30, font=APP_FONT, fg_color="#444444"); btn_mute.configure(command=lambda c=ch, b=btn_mute: self.toggle_mute(c, b)); btn_mute.pack(side="left", padx=1); self.ch_ui_refs[ch.name]['btn_mute'] = btn_mute
            self.widget_to_lfo_target[str(btn_mute)] = (f"ch:{i}:mute", f"{ch.name} Mute")
            btn_solo = ctk.CTkButton(btn_frame, text="S", width=30, font=APP_FONT, fg_color="#444444"); btn_solo.pack(side="left", padx=1); btn_solo.bind("<ButtonPress-1>", lambda e, c=ch, b=btn_solo: self.solo_press(c, b)); btn_solo.bind("<ButtonRelease-1>", lambda e, c=ch, b=btn_solo: self.solo_release(c, b)); btn_solo.bind("<Double-Button-1>", lambda e, c=ch, b=btn_solo: self.solo_double(c, b)); self.ch_ui_refs[ch.name]['btn_solo'] = btn_solo
            self.widget_to_lfo_target[str(btn_solo)] = (f"ch:{i}:solo", f"{ch.name} Solo")
            
            btn_cycl = ctk.CTkButton(btn_frame, text="Cycle", width=35, font=APP_FONT, fg_color="#2ECC71", hover_color="#27AE60", text_color="#000000")
            btn_cycl.bind("<ButtonPress-1>", lambda e, c=ch, b=btn_cycl: self._on_cyc_press(c, b)); btn_cycl.bind("<ButtonRelease-1>", lambda e, c=ch, b=btn_cycl: self._on_cyc_release(c, b)); btn_cycl.bind("<Button-3>", lambda e, c=ch: self.cycle_drumb(c, False)); btn_cycl.pack(side="left", padx=1)
            self.widget_to_lfo_target[str(btn_cycl)] = (f"ch:{i}:cyc", f"{ch.name} Cycle")
            
            self.ch_ui_refs[ch.name]['vol'] = self.add_ch_sl(col, "Vol", 0, 1, ch.vol, lambda v, c=ch: self.handle_ch_slider(c, 'vol', v), f"ch:{i}:vol", f"{ch.name} Vol")
            self.ch_ui_refs[ch.name]['pan'] = self.add_ch_sl(col, "Pan", -1, 1, ch.pan, lambda v, c=ch: self.handle_ch_slider(c, 'pan', v), f"ch:{i}:pan", f"{ch.name} Pan")
            self.ch_ui_refs[ch.name]['pitch'] = self.add_ch_sl(col, "Pitch", -24, 24, ch.pitch, lambda v, c=ch: self.handle_ch_slider(c, 'pitch', v), f"ch:{i}:pitch", f"{ch.name} Pitch", is_int=True)
            self.ch_ui_refs[ch.name]['sample'] = self.add_ch_range_sl(col, "Sample", 0, 300, ch.sample_start, ch.sample_end, lambda v1, v2, c=ch: self.handle_ch_slider(c, 'sample', v1, True, v2), f"ch:{i}:sam_s", f"ch:{i}:sam_e", f"{ch.name} Sample")
            
            self.ch_ui_refs[ch.name]['fx1'] = self.add_ch_sl(col, "FX 1", 0, 1, ch.fx1, lambda v, c=ch: self.handle_ch_slider(c, 'fx1', v), f"ch:{i}:fx1", f"{ch.name} FX1")
            self.ch_ui_refs[ch.name]['fx2'] = self.add_ch_sl(col, "FX 2", 0, 1, ch.fx2, lambda v, c=ch: self.handle_ch_slider(c, 'fx2', v), f"ch:{i}:fx2", f"{ch.name} FX2")
            self.ch_ui_refs[ch.name]['fx3'] = self.add_ch_sl(col, "FX 3", 0, 1, ch.fx3, lambda v, c=ch: self.handle_ch_slider(c, 'fx3', v), f"ch:{i}:fx3", f"{ch.name} FX3")
            self.ch_ui_refs[ch.name]['euc'] = self.add_ch_sl(col, "Euc", 0, self.engine.steps, ch.euclid_k, lambda v, c=ch: self.handle_ch_slider(c, 'euc', v), f"ch:{i}:euc", f"{ch.name} Euc", is_int=True)

        self.seq_frame = ctk.CTkScrollableFrame(mixer_container, fg_color="#141414", orientation="vertical"); self.seq_frame.pack(fill="both", expand=True)

        for i in range(3):
            f_box = ctk.CTkFrame(self.fx_frame, fg_color="#1A1A1A"); f_box.pack(fill="x", padx=5, pady=5)
            header_f = ctk.CTkFrame(f_box, fg_color="transparent"); header_f.pack(fill="x", padx=5, pady=(5,0))
            ctk.CTkLabel(header_f, text=f"FX{i+1}", font=APP_FONT_BOLD, text_color="#AAAAAA").pack(side="left")
            opt = ctk.CTkOptionMenu(header_f, values=list(FX_DEFS.keys()), font=APP_FONT, width=120, height=24, command=lambda v, idx=i: self.on_fx_type_change(idx, v))
            opt.set(self.engine.fx_buses[i].fx_type); opt.pack(side="right")

            knobs_f = ctk.CTkFrame(f_box, fg_color="transparent"); knobs_f.pack(fill="x", padx=5, pady=5)
            sl_refs =[]
            for j in range(3):
                k_frame = ctk.CTkFrame(knobs_f, fg_color="transparent"); k_frame.pack(side="left", expand=True)
                def on_change(v, b=i, p=j): self.engine.fx_buses[b].set_param(p+1, v); self.update_fx_lbl_text(b, p, v)
                knob = CTkKnob(k_frame, width=40, height=40, from_=0, to=1, command=on_change); knob.pack(pady=(2,0))
                lbl = ctk.CTkLabel(k_frame, text="", font=("Trebuchet MS", 10)); lbl.pack()
                self.bind_knob_events(knob, False, on_change, lbl, f"fx:{i}:p{j+1}", f"Master FX{i+1} P{j+1}", j, i)
                sl_refs.append({'knob': knob, 'lbl': lbl, 'frame': k_frame})
            self.fx_ui_refs.append({'opt': opt, 'sliders': sl_refs}); self.update_fx_ui_labels(i)

        lfo_container = ctk.CTkFrame(self.fx_frame, fg_color="transparent")
        lfo_container.pack(fill="x", pady=10)
        lfo_cols =["#00FFFF", "#FF00FF", "#FFFF00", "#00FF00"]
        for row in range(2):
            row_f = ctk.CTkFrame(lfo_container, fg_color="transparent")
            row_f.pack(fill="x", pady=2)
            for col in range(2):
                i = row * 2 + col
                f_box = ctk.CTkFrame(row_f, fg_color="#1A1A1A")
                f_box.pack(side="left", fill="both", expand=True, padx=2)
                
                h_f = ctk.CTkFrame(f_box, fg_color="transparent"); h_f.pack(fill="x", padx=2, pady=2)
                ctk.CTkLabel(h_f, text=f"LFO {i+1}", font=("Trebuchet MS", 10, "bold"), text_color=lfo_cols[i]).pack(side="left")
                
                lbl_targ = ctk.CTkLabel(h_f, text="[None]", font=("Trebuchet MS", 9), text_color="#AAAAAA")
                lbl_targ.pack(side="left", padx=2)
                ctk.CTkButton(h_f, text="❌", width=16, height=16, font=("Arial", 10), fg_color="#6B1D1D", command=lambda idx=i: self.clear_lfo_target(idx)).pack(side="right")
                
                r2 = ctk.CTkFrame(f_box, fg_color="transparent"); r2.pack(fill="x", padx=2, pady=2)
                opt_shp = ctk.CTkOptionMenu(r2, values=["Sine", "Triangle", "Square", "Random"], width=60, height=20, font=("Trebuchet MS", 9), command=lambda v, idx=i: setattr(self.engine.lfos[idx], 'shape', v))
                opt_shp.set("Sine"); opt_shp.pack(side="left", padx=1)
                btn_sync = ctk.CTkButton(r2, text="BPM 🔓", width=40, height=20, font=("Trebuchet MS", 9), fg_color="#333", command=lambda idx=i: self.toggle_lfo_sync(idx))
                btn_sync.pack(side="left", padx=1)
                
                r3 = ctk.CTkFrame(f_box, fg_color="transparent"); r3.pack(fill="x", padx=2, pady=2)
                opt_rate = ctk.CTkOptionMenu(r3, values=["8 Bar", "4 Bar", "2 Bar", "1 Bar", "1/2", "1/4", "1/8", "1/16"], width=50, height=20, font=("Trebuchet MS", 9), command=lambda v, idx=i: setattr(self.engine.lfos[idx], 'rate_sync', v))
                opt_rate.set("1/4")
                
                rate_k = CTkKnob(r3, width=24, height=24, from_=0.066, to=20.0, command=lambda v, idx=i: setattr(self.engine.lfos[idx], 'rate_hz', v))
                rate_k.set(1.0); rate_k.pack(side="left", padx=1)
                
                depth_k = CTkKnob(r3, width=24, height=24, from_=0.0, to=1.0, command=lambda v, idx=i: setattr(self.engine.lfos[idx], 'depth', v))
                depth_k.set(0.5); depth_k.pack(side="left", padx=1)
                ctk.CTkLabel(r3, text="Depth", font=("Trebuchet MS", 8)).pack(side="left", padx=1)
                
                led = ctk.CTkFrame(r3, width=10, height=10, corner_radius=5, fg_color="#222"); led.pack(side="right", padx=2)
                
                self.lfo_ui_refs.append({'lbl_target': lbl_targ, 'opt_shp': opt_shp, 'btn_sync': btn_sync, 'opt_rate': opt_rate, 'rate_k': rate_k, 'depth_k': depth_k, 'led': led})

        self.build_grid_ui()

    def clear_lfo_target(self, idx):
        self.engine.lfos[idx].target_id = None; self.engine.lfos[idx].target_name = None
        self.lfo_ui_refs[idx]['lbl_target'].configure(text="[None]")

    def toggle_lfo_sync(self, idx):
        lfo = self.engine.lfos[idx]
        lfo.sync = not lfo.sync
        btn = self.lfo_ui_refs[idx]['btn_sync']
        btn.configure(text="BPM 🔒" if lfo.sync else "BPM 🔓", fg_color="#2ECC71" if lfo.sync else "#333")
        if lfo.sync:
            self.lfo_ui_refs[idx]['rate_k'].pack_forget()
            self.lfo_ui_refs[idx]['opt_rate'].pack(side="left", padx=1)
        else:
            self.lfo_ui_refs[idx]['opt_rate'].pack_forget()
            self.lfo_ui_refs[idx]['rate_k'].pack(side="left", padx=1)

    def _on_cyc_press(self, ch, btn):
        setattr(btn, '_pressed', True); setattr(btn, '_held', False)
        t = threading.Timer(2.0, lambda: self._on_cyc_hold(ch, btn)); t.start(); setattr(btn, '_timer', t)

    def _on_cyc_release(self, ch, btn):
        if getattr(btn, '_pressed', False):
            setattr(btn, '_pressed', False)
            if hasattr(btn, '_timer'): btn._timer.cancel()
            if not getattr(btn, '_held', False): self.cycle_drumb(ch)

    def _on_cyc_hold(self, ch, btn):
        if getattr(btn, '_pressed', False):
            setattr(btn, '_held', True); btn.configure(text="Wait...")
            self.rescan_single_channel(ch, btn)

    def handle_ch_slider(self, ch, param, val, is_range=False, val2=None):
        targets =[c for c in self.engine.channels if not c.locked] if self.ctrl_pressed else [ch]
        if ch not in targets and not ch.locked: targets.append(ch)
        
        for c in targets:
            if is_range:
                c.sample_start, c.sample_end = val, val2; c.render()
                self.ch_ui_refs[c.name]['sample'][0].set(val, val2); self.ch_ui_refs[c.name]['sample'][1].configure(text=f"Sample: {int(val)}-{int(val2)}")
            elif param == 'euc':
                self.apply_euclidean(c, int(val), refresh_ui=False, randomize=False)
                self.ch_ui_refs[c.name]['euc'][0].set(int(val)); self.ch_ui_refs[c.name]['euc'][1].configure(text=f"Euc: {int(val)}")
            else:
                setattr(c, param, val)
                if param in ['pitch']: c.render()
                ui_t = self.ch_ui_refs[c.name][param]
                ui_t[0].set(val); l_txt = param.capitalize() if "fx" not in param else param.upper().replace("X", "X ")
                ui_t[1].configure(text=f"{l_txt}: {val:.2f}" if param != 'pitch' else f"{l_txt}: {int(val)}")
        if param == 'euc': self.refresh_grid_ui()

    def bind_knob_events(self, knob, is_int, command, lbl, t_id, t_name, p_idx=0, bus_idx=0, global_def=0.0):
        self.widget_to_lfo_target[str(knob)] = (t_id, t_name)
        self.widget_to_lfo_target[str(knob.canvas)] = (t_id, t_name)
        self.widget_to_lfo_target[str(lbl)] = (t_id, t_name)
        self.target_updater_map[t_id] = lambda v=knob: v.set(getattr(self.engine, "m_"+t_id.split(":")[1]) if t_id.startswith("gl") else getattr(self.engine.fx_buses[int(t_id.split(":")[1])], "m_"+t_id.split(":")[2])) if not v.is_dragging else None
        
        def on_change(v):
            val = int(v) if is_int else v; command(val)
        def reset(e):
            self.save_state()
            d = FX_DEFS[self.engine.fx_buses[bus_idx].fx_type][p_idx][3] if hasattr(knob, 'is_fx_rack') else global_def
            knob.set(d); on_change(d)
        def manual_entry(e):
            prompt = FX_DEFS[self.engine.fx_buses[bus_idx].fx_type][p_idx][0] if hasattr(knob, 'is_fx_rack') else "Manual Entry"
            res = ctk.CTkInputDialog(text=f"Enter {prompt} ({knob.min_val} to {knob.max_val}):", title="Entry").get_input()
            if res is not None:
                self.save_state()
                try: v = float(res)
                except ValueError: v = random.uniform(knob.min_val, knob.max_val)
                v = max(knob.min_val, min(knob.max_val, v))
                v = round(v) if is_int else v; knob.set(v); on_change(v)
        def wheel(e):
            if (dir_step := (1 if e.delta > 0 else -1) if hasattr(e, 'delta') and e.delta != 0 else (1 if e.num == 4 else -1)):
                self.save_state(); step = max(1, int((knob.max_val - knob.min_val) / 50.0)) if is_int else (knob.max_val - knob.min_val) / 50.0
                new_val = max(knob.min_val, min(knob.max_val, knob.get() + (dir_step * step)))
                new_val = int(round(new_val)) if is_int else new_val; knob.set(new_val); on_change(new_val)
        lbl.bind("<Double-Button-1>", manual_entry)
        for w in[knob, knob.canvas]:
            w.bind("<Double-Button-1>", reset); w.bind("<MouseWheel>", wheel); w.bind("<Button-4>", wheel); w.bind("<Button-5>", wheel)

    def add_global_knob(self, parent, label, min_val, max_val, default, command, t_id, is_int=False):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(side="left", padx=5)
        def on_change(v):
            val = int(v) if is_int else v; lbl.configure(text=f"{label}\n{val:.2f}" if not is_int else f"{label}\n{val}"); command(val)
        knob = CTkKnob(f, width=40, height=40, from_=min_val, to=max_val, command=on_change); knob.set(default); knob.pack()
        lbl = ctk.CTkLabel(f, text=f"{label}\n{default:.2f}" if not is_int else f"{label}\n{int(default)}", font=("Trebuchet MS", 10)); lbl.pack()
        self.bind_knob_events(knob, is_int, on_change, lbl, t_id, label, global_def=default)
        self.g_ui_refs[label] = knob; self.g_lbl_refs[label] = lbl

    def add_ch_range_sl(self, parent, label, min_val, max_val, def_start, def_end, command, t_id_start, t_id_end, t_name):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill="x", padx=2, pady=0)
        lbl = ctk.CTkLabel(f, text=f"{label}: {int(def_start)}-{int(def_end)}", font=("Trebuchet MS", 11)); lbl.pack(anchor="w", pady=0)
        def on_change(v1, v2): lbl.configure(text=f"{label}: {int(v1)}-{int(v2)}"); command(v1, v2)
        sl = CTkRangeSlider(f, width=130, height=20, from_=min_val, to=max_val, command=on_change); sl.set(def_start, def_end); sl.pack(anchor="w")
        for w in [sl, sl.canvas, lbl]:
            self.widget_to_lfo_target[str(w)] = (t_id_end, f"{t_name} End")
        
        ch_idx = int(t_id_start.split(":")[1])
        self.target_updater_map[t_id_start] = lambda s=sl, c=ch_idx: s.set(self.engine.channels[c].m_sample_start, self.engine.channels[c].m_sample_end) if not s.is_dragging else None
        self.target_updater_map[t_id_end] = lambda s=sl, c=ch_idx: s.set(self.engine.channels[c].m_sample_start, self.engine.channels[c].m_sample_end) if not s.is_dragging else None
        return sl, lbl

    def add_ch_sl(self, parent, label, min_val, max_val, default, command, t_id, t_name, is_int=False):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill="x", padx=2, pady=0)
        lbl = ctk.CTkLabel(f, text=f"{label}: {default:.2f}" if not is_int else f"{label}: {int(default)}", font=("Trebuchet MS", 11)); lbl.pack(anchor="w", pady=0)
        def on_change(v):
            val = int(v) if is_int else v; lbl.configure(text=f"{label}: {val:.2f}" if not is_int else f"{label}: {val}"); command(val)
        sl = ctk.CTkSlider(f, from_=min_val, to=max_val, width=130, command=on_change); sl.set(default); sl.pack(anchor="w")
        
        ch_idx, param = int(t_id.split(":")[1]), t_id.split(":")[2]
        self.target_updater_map[t_id] = lambda s=sl, c=ch_idx, p=param: s.set(getattr(self.engine.channels[c], "m_"+p))
        
        for w in [sl, sl._canvas, lbl]: self.widget_to_lfo_target[str(w)] = (t_id, t_name)
        
        def reset(e): self.save_state(); sl.set(default); on_change(default)
        def manual_entry(e):
            res = ctk.CTkInputDialog(text=f"Enter {label} ({min_val} to {max_val}):", title="Entry").get_input()
            if res is not None:
                self.save_state()
                try: v = float(res)
                except ValueError: v = random.uniform(min_val, max_val)
                v = max(min_val, min(max_val, v))
                v = round(v) if is_int else v; sl.set(v); on_change(v)
        def wheel(e):
            if (dir_step := (1 if e.delta > 0 else -1) if hasattr(e, 'delta') and e.delta != 0 else (1 if e.num == 4 else -1)):
                self.save_state(); step = max(1, int((max_val - min_val)/50.0)) if is_int else (max_val - min_val)/50.0
                new_val = max(min_val, min(max_val, sl.get() + (dir_step * step)))
                new_val = int(round(new_val)) if is_int else new_val; sl.set(new_val); on_change(new_val)
        lbl.bind("<Double-Button-1>", manual_entry)
        for w in[sl, sl._canvas]: w.bind("<Double-Button-1>", reset); w.bind("<MouseWheel>", wheel); w.bind("<Button-4>", wheel); w.bind("<Button-5>", wheel)
        return sl, lbl

    def on_fx_type_change(self, bus_idx, new_type):
        self.save_state(); self.engine.fx_buses[bus_idx].set_type(new_type); self.update_fx_ui_labels(bus_idx)

    def update_fx_ui_labels(self, bus_idx):
        bus = self.engine.fx_buses[bus_idx]; defs = FX_DEFS[bus.fx_type]; ui = self.fx_ui_refs[bus_idx]
        ui['opt'].set(bus.fx_type)
        for j in range(3):
            p_def = defs[j]; sl_dict = ui['sliders'][j]
            if not p_def[0]: sl_dict['frame'].pack_forget()
            else:
                sl_dict['frame'].pack(side="left", expand=True)
                sl_dict['knob'].is_fx_rack = True; sl_dict['knob'].configure_range(p_def[1], p_def[2])
                val = getattr(bus, f"p{j+1}"); sl_dict['knob'].set(val); self.update_fx_lbl_text(bus_idx, j, val)

    def update_fx_lbl_text(self, bus_idx, p_idx, val):
        bus = self.engine.fx_buses[bus_idx]; p_name = FX_DEFS[bus.fx_type][p_idx][0]
        if p_name:
            if bus.fx_type == "Resonant LPF" and p_idx == 0: self.fx_ui_refs[bus_idx]['sliders'][p_idx]['lbl'].configure(text=f"{p_name}\n{int(val)} Hz")
            else: self.fx_ui_refs[bus_idx]['sliders'][p_idx]['lbl'].configure(text=f"{p_name}\n{val:.2f}")

    def open_options(self):
        w = ctk.CTkToplevel(self); w.title("Options"); w.geometry("300x150"); w.transient(self)
        ctk.CTkLabel(w, text="Settings", font=APP_FONT_BOLD).pack(pady=10)
        ctk.CTkCheckBox(w, text="More random sampling (Old Method)", font=APP_FONT, variable=self.use_old_extraction).pack(pady=10)

    def sync_ui_to_engine(self):
        for p, k, is_int in[('Vol','global_vol',False),('FX1 Send','master_fx1',False),('FX2 Send','master_fx2',False),('FX3 Send','master_fx3',False),('Pitch','global_pitch',True),('Swing','swing',False)]:
            val = getattr(self.engine, k); self.g_ui_refs[p].set(val)
            self.g_lbl_refs[p].configure(text=f"{p}\n{val:.2f}" if not is_int else f"{p}\n{int(val)}")
        self.bpm_widget.set_val(self.engine.bpm); self.pat_var.set(str(self.engine.current_pattern)); self.step_var.set(str(self.engine.steps))
        
        for ch in self.engine.channels:
            r = self.ch_ui_refs[ch.name]
            r['btn_mute'].configure(fg_color="#CC0000" if ch.mute else "#444444")
            r['btn_lock'].configure(text="🔒", text_color="#E74C3C") if ch.locked else r['btn_lock'].configure(text="🔓", text_color="#FFFFFF")
            if ch.solo_locked: r['btn_solo'].configure(fg_color="#CCCC00")
            elif ch.solo: r['btn_solo'].configure(fg_color="#FF9900")
            else: r['btn_solo'].configure(fg_color="#444444")
            
            for p, attr, is_int in[('vol','vol',False),('pan','pan',False),('pitch','pitch',True),('fx1','fx1',False),('fx2','fx2',False),('fx3','fx3',False),('euc','euclid_k',True)]:
                sl, lbl = r[p]; v = getattr(ch, attr); sl.set(v)
                label_text = p.capitalize() if "fx" not in p else p.upper().replace("X", "X ")
                lbl.configure(text=f"{label_text}: {v:.2f}" if not is_int else f"{label_text}: {int(v)}")
            r['euc'][0].configure(to=self.engine.steps)
            r['sample'][0].set(ch.sample_start, ch.sample_end); r['sample'][1].configure(text=f"Sample: {int(ch.sample_start)}-{int(ch.sample_end)}")
        for i in range(3): self.update_fx_ui_labels(i)
        self.build_grid_ui() if len(self.grid_buttons) != 8 * self.engine.steps else self.refresh_grid_ui()

    def reset_all_params(self, e=None):
        self.save_state()
        self.engine.apply_state({
            'bpm': 120, 'steps': 16, 'pat': 0, 'swing': 0.0, 'gvol': 0.8, 'gpitch': 0, 'm_fx1': 0.0, 'm_fx2': 0.0, 'm_fx3': 0.0,
            'ch':[{'idx': 0, 'mute': False, 'solo_l': False, 'locked': False, 'vol': 0.8, 'pan': 0.0, 'pitch': 0, 's_start': 0, 's_end': 200, 'fx1': 0.0, 'fx2': 0.0, 'fx3': 0.0, 'euc': 0, 'seq': [[False]*32 for _ in range(4)], 'ps':[{'vol':0.8, 'pan':0.0, 'pitch':0, 's_start':0, 's_end':200, 'fx1':0.0, 'fx2':0.0, 'fx3':0.0, 'euc':0} for _ in range(4)]} for _ in range(8)],
            'fx':[{'type': 'Reverb', 'p1': 0.5, 'p2': 0.5, 'p3': 0.0}, {'type': 'Resonant LPF', 'p1': 20000.0, 'p2': 0.0, 'p3': 0.0}, {'type': 'Compressor', 'p1': 0.1, 'p2': 4.0, 'p3': 0.05}]
        }); self.sync_ui_to_engine(); self.lbl_status.configure(text="All parameters reset.")

    def load_file(self):
        if fp := filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")]):
            self.current_audio_file = fp; self.lbl_status.configure(text="Extracting drumbs... Please wait.")
            threading.Thread(target=self.process_audio_file, args=(fp, False), daemon=True).start()

    def rescan_file(self):
        if self.current_audio_file:
            self.lbl_status.configure(text="Rescanning for fresh drumbs...")
            threading.Thread(target=self.process_audio_file, args=(self.current_audio_file, True), daemon=True).start()
        else: self.lbl_status.configure(text="Load a file first before rescanning!")

    def rescan_single_channel(self, ch, btn):
        if not self.current_audio_file:
            self.after(0, lambda:[btn.configure(text="Cycle"), self.lbl_status.configure(text="No file loaded to rescan!")]); return
        def task():
            try:
                ch.samples = self.extractor.rescan_single_channel(self.current_audio_file, ch.name)
                ch.current_sample_idx = 0; ch.render()
                self.after(0, lambda: self.lbl_status.configure(text=f"Rescanned new samples for {ch.name}"))
            except Exception as e: self.after(0, lambda e=e: self.lbl_status.configure(text=f"Error: {e}"))
            finally: self.after(0, lambda: btn.configure(text="Cycle"))
        threading.Thread(target=task, daemon=True).start()

    def process_audio_file(self, filepath, is_rescan=False):
        is_initial = not self.first_load_done
        try:
            drumbs = self.extractor.extract(filepath, old_way=self.use_old_extraction.get(), is_rescan=is_rescan)
            for ch in self.engine.channels:
                if ch.name in drumbs: ch.samples = drumbs[ch.name]; ch.render()
            
            def _update_gui():
                if is_initial:
                    self.first_load_done = True; self.save_state()
                    self.refresh_grid_ui()
                self.lbl_status.configure(text=f"Extraction complete!")
                
            self.after(0, _update_gui)
        except Exception as e: self.after(0, lambda e=e: self.lbl_status.configure(text=f"Error: {str(e)}"))

    def toggle_play(self):
        self.engine.is_playing = not self.engine.is_playing
        if self.engine.is_playing: 
            self.engine.step_counter = 0; self.engine.samples_until_next_step = 0.0
            self.engine.master_step_counter = 0; self.engine.master_samples_until_next_step = 0.0

    def toggle_record(self):
        if not self.engine.is_recording:
            self.engine.record_buffer =[]; self.engine.is_recording = True; self.btn_rec.configure(fg_color="#CC0000", text="STOP REC")
        else:
            self.engine.is_recording = False; self.btn_rec.configure(fg_color="transparent", text="RECORD")
            if self.engine.record_buffer:
                audio_data = np.concatenate(self.engine.record_buffer, axis=0)
                if fp := filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")]): wavfile.write(fp, SAMPLE_RATE, audio_data); self.lbl_status.configure(text=f"Recording saved.")

    def unsolo_all(self, e=None):
        self.solo_history.append({ch.name: (ch.solo, ch.solo_locked) for ch in self.engine.channels})
        for ch in self.engine.channels:
            ch.solo = False; ch.solo_locked = False
            self.ch_ui_refs[ch.name]['btn_solo'].configure(fg_color="#444444")
            
    def undo_unsolo_all(self, e=None):
        if not self.solo_history: return
        state = self.solo_history.pop()
        for ch in self.engine.channels:
            if ch.name in state:
                ch.solo, ch.solo_locked = state[ch.name]
                if ch.solo_locked: self.ch_ui_refs[ch.name]['btn_solo'].configure(fg_color="#CCCC00")
                elif ch.solo: self.ch_ui_refs[ch.name]['btn_solo'].configure(fg_color="#FF9900")
                else: self.ch_ui_refs[ch.name]['btn_solo'].configure(fg_color="#444444")
            
    def unmute_all(self, e=None):
        self.mute_history.append({ch.name: ch.mute for ch in self.engine.channels})
        for ch in self.engine.channels:
            ch.mute = False
            self.ch_ui_refs[ch.name]['btn_mute'].configure(fg_color="#444444")

    def undo_unmute_all(self, e=None):
        if not self.mute_history: return
        state = self.mute_history.pop()
        for ch in self.engine.channels:
            if ch.name in state:
                ch.mute = state[ch.name]
                self.ch_ui_refs[ch.name]['btn_mute'].configure(fg_color="#CC0000" if ch.mute else "#444444")

    def toggle_mute(self, ch, btn): ch.mute = not ch.mute; btn.configure(fg_color="#CC0000" if ch.mute else "#444444")
    def toggle_lock(self, ch, btn): 
        ch.locked = not ch.locked
        btn.configure(text="🔒", text_color="#E74C3C") if ch.locked else btn.configure(text="🔓", text_color="#FFFFFF")

    def solo_press(self, ch, btn): ch.solo = True; btn.configure(fg_color="#FF9900")
    def solo_release(self, ch, btn):
        if not ch.solo_locked: ch.solo = False; btn.configure(fg_color="#444444")
    def solo_double(self, ch, btn):
        ch.solo_locked = not ch.solo_locked; ch.solo = ch.solo_locked; btn.configure(fg_color="#CCCC00" if ch.solo_locked else "#444444")

    def cycle_drumb(self, ch, forward=True): ch.step_sample(forward); self.lbl_status.configure(text=f"{ch.name} -> drumb {ch.current_sample_idx + 1}/15")

    def get_weighted_euc_val(self, steps):
        r = random.random(); q1, q2, q3 = math.floor(steps * 0.25), math.floor(steps * 0.50), math.floor(steps * 0.75)
        if r < 0.50: return random.randint(0, q1)
        elif r < 0.75: return random.randint(q1 + 1, q2)
        elif r < 0.90: return random.randint(q2 + 1, q3)
        else: return random.randint(q3 + 1, steps)

    def apply_euclidean(self, ch, hits, refresh_ui=True, randomize=False):
        ch.euclid_k = hits; pat = self.engine.current_pattern; steps = self.engine.steps
        if hits == 0: 
            for i in range(32): ch.sequence[pat][i] = False
        else:
            hits = min(hits, steps)
            for i in range(32): ch.sequence[pat][i] = False
            for i in range(steps): ch.sequence[pat][i] = ((i * hits) % steps) < hits
            
            if randomize:
                if ch.sequence[pat][0] and random.random() < 0.50:
                    ch.sequence[pat][0] = False
                    
                if random.random() < 0.20:
                    curr_seq = ch.sequence[pat][:steps]
                    new_seq = curr_seq[-2:] + curr_seq[:-2]
                    ch.sequence[pat][:steps] = new_seq
                    
        if refresh_ui: self.refresh_grid_ui()

    def roll_single_euc(self, ch):
        if ch.locked: return
        self.save_state(); val = self.get_weighted_euc_val(self.engine.steps)
        self.apply_euclidean(ch, val, randomize=True); self.ch_ui_refs[ch.name]['euc'][0].set(val); self.ch_ui_refs[ch.name]['euc'][1].configure(text=f"Euc: {val}")

    def rand_euc(self):
        for ch in self.engine.channels:
            if not ch.locked:
                val = self.get_weighted_euc_val(self.engine.steps)
                self.apply_euclidean(ch, val, randomize=True); self.ch_ui_refs[ch.name]['euc'][0].set(val); self.ch_ui_refs[ch.name]['euc'][1].configure(text=f"Euc: {val}")

    def reset_rand_sam(self, e=None):
        self.save_state()
        for ch in self.engine.channels:
            if not ch.locked: self.handle_ch_slider(ch, 'sample', 0, True, 200)

    def rand_sam(self):
        for ch in self.engine.channels:
            if not ch.locked:
                s_end = random.uniform(100, 300); s_start = random.uniform(0, max(0, s_end - 50))
                self.handle_ch_slider(ch, 'sample', s_start, True, s_end)

    def rand_cycl(self):
        for ch in self.engine.channels: 
            if not ch.locked: self.cycle_drumb(ch)

    def rand_pan(self):
        for ch in self.engine.channels:
            if not ch.locked:
                r = random.random()
                if r < 0.50: p = random.uniform(-0.2, 0.2)
                elif r < 0.75: p = random.uniform(0.2, 0.5) * random.choice([-1, 1])
                elif r < 0.92: p = random.uniform(0.5, 0.8) * random.choice([-1, 1])
                else: p = random.uniform(0.8, 1.0) * random.choice([-1, 1])
                self.handle_ch_slider(ch, 'pan', p)
            
    def rand_pitch(self):
        for ch in self.engine.channels:
            if not ch.locked: self.handle_ch_slider(ch, 'pitch', random.randint(-24, 24))
            
    def rand_all(self): self.rand_cycl(); self.rand_euc(); self.rand_sam(); self.rand_pan(); self.rand_pitch()

    def change_pattern(self, val):
        self.save_state()
        target_pat, curr_pat = int(val), self.engine.current_pattern
        for ch in self.engine.channels: ch.save_pattern_state(curr_pat)
        if target_pat != curr_pat and all(not any(ch.sequence[target_pat]) for ch in self.engine.channels):
            for ch in self.engine.channels: 
                ch.sequence[target_pat] =[s for s in ch.sequence[curr_pat]]
                ch.pattern_settings[target_pat] = copy.deepcopy(ch.pattern_settings[curr_pat])
        self.engine.current_pattern = target_pat
        for ch in self.engine.channels: ch.load_pattern_state(target_pat)
        self.sync_ui_to_engine()

    def change_steps(self, val):
        val = int(val); old_steps = self.engine.steps; self.save_state()
        if val > old_steps:
            for ch in self.engine.channels:
                for pat in range(4):
                    old_seq = ch.sequence[pat][:old_steps]
                    for i in range(old_steps, val): ch.sequence[pat][i] = old_seq[i % old_steps]
        self.engine.steps = val; self.build_grid_ui()
        for ch in self.engine.channels:
            self.ch_ui_refs[ch.name]['euc'][0].configure(to=val)
            if ch.euclid_k > val:
                self.handle_ch_slider(ch, 'euc', val)

    def toggle_step(self, ch_idx, step_idx, state):
        self.engine.channels[ch_idx].sequence[self.engine.current_pattern][step_idx] = state
        self.grid_buttons[(ch_idx, step_idx)].configure(fg_color=ROW_COLORS[ch_idx] if state else "transparent")

    def on_paint(self, e, state):
        try:
            if (w := self.winfo_containing(e.x_root, e.y_root)) and str(w) in self.widget_to_step:
                ch_idx, step_idx = self.widget_to_step[str(w)]
                if self.engine.channels[ch_idx].sequence[self.engine.current_pattern][step_idx] != state:
                    self.engine.channels[ch_idx].sequence[self.engine.current_pattern][step_idx] = state
                    self.grid_buttons[(ch_idx, step_idx)].configure(fg_color=ROW_COLORS[ch_idx] if state else "transparent")
        except Exception: pass

    def build_grid_ui(self):
        for widget in self.seq_frame.winfo_children(): widget.destroy()
        self.widget_to_step.clear(); self.grid_buttons.clear()
        
        steps = self.engine.steps; pat = self.engine.current_pattern
        inner_container = ctk.CTkFrame(self.seq_frame, fg_color="transparent"); inner_container.pack(fill="x", anchor="n")
        
        for i, ch in enumerate(self.engine.channels):
            row_frame = ctk.CTkFrame(inner_container, fg_color="transparent"); row_frame.pack(fill="x", pady=1) 
            title_f = ctk.CTkFrame(row_frame, fg_color="transparent", width=90, height=32); title_f.pack(side="left", padx=5); title_f.pack_propagate(False)
            ctk.CTkLabel(title_f, text=ch.name, anchor="e", font=APP_FONT_BOLD).pack(side="left", padx=2, fill="x", expand=True)
            ctk.CTkButton(title_f, text="🎲", width=22, height=22, font=("Arial", 12), fg_color="#333333", hover_color="#555555", command=lambda c=ch: self.roll_single_euc(c)).pack(side="right")
            
            for j in range(steps):
                bg_col = "#151515" if (j // 4) % 2 == 0 else "#252525" 
                step_bg = ctk.CTkFrame(row_frame, fg_color=bg_col, corner_radius=4, height=32, width=32); step_bg.pack(side="left", padx=1, expand=True, fill="both"); step_bg.pack_propagate(False)
                state = ch.sequence[pat][j]
                btn = ctk.CTkButton(step_bg, text="", corner_radius=2, fg_color=ROW_COLORS[i] if state else "transparent", hover_color=ROW_COLORS[i], border_width=1, border_color="#333333", command=lambda c=i, s=j: self.toggle_step(c, s, not self.engine.channels[c].sequence[self.engine.current_pattern][s]))
                btn.place(relx=0.5, rely=0.5, relwidth=0.8, relheight=0.8, anchor="center")
                self.grid_buttons[(i, j)] = btn
                for w in[str(btn), str(btn._canvas), str(step_bg)]: 
                    self.widget_to_step[w] = (i, j)
                    self.widget_to_lfo_target[w] = (f"grid:{i}:{j}", f"{ch.name} Step {j+1}")

    def refresh_grid_ui(self):
        pat = self.engine.current_pattern
        for i, ch in enumerate(self.engine.channels):
            for j in range(self.engine.steps):
                state = ch.m_sequence[pat][j] if getattr(ch, 'm_sequence', None) else ch.sequence[pat][j]
                if (i, j) in self.grid_buttons:
                    col = ROW_COLORS[i] if state else "transparent"
                    btn = self.grid_buttons[(i, j)]
                    if btn.cget("fg_color") != col:
                        btn.configure(fg_color=col)

if __name__ == "__main__":
    try:
        engine = AudioEngine()
        app = DrumMachineApp(engine)
        app.mainloop()
    except Exception as e: print(f"Error starting application: {e}")
    finally:
        if 'engine' in locals(): engine.stream.stop(); engine.stream.close()