import os, copy, math, random, threading, time, warnings
from collections import deque
import numpy as np
import librosa
import sounddevice as sd
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import scipy.io.wavfile as wavfile
from pedalboard import Pedalboard, Reverb, Delay, Distortion, Compressor, LowpassFilter, Chorus
try:
    from pedalboard import LadderFilter
    HAS_LADDER = True
except ImportError: HAS_LADDER = False

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SAMPLE_RATE, BLOCK_SIZE = 44100, 1024
CHANNELS = ["Kick", "Snare", "Closed Hat", "Open Hat", "Hi Tom", "Low Tom", "Crash", "Ride"]
ROW_COLORS = ["#E74C3C", "#F39C12", "#3498DB", "#2980B9", "#9B59B6", "#8E44AD", "#1ABC9C", "#BDC3C7"]
FX_COLORS = ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71", "#9B59B6", "#1ABC9C", "#D35400", "#8E44AD"]
APP_FONT, APP_FONT_BOLD = ("Trebuchet MS", 12), ("Trebuchet MS", 12, "bold")

STUTTER_OPTS = [(4, "1/4"), (6, "1/4T"), (8, "1/8"), (12, "1/8T"), (16, "1/16"), (24, "1/16T"), (32, "1/32"), (48, "1/32T"), (64, "1/64"), (96, "1/64T")]

FX_DEFS = {
    "Reverb": [("Size",0.0,1.0,0.2), ("Length",0.0,1.0,0.2), ("PreDly",0.0,0.5,0.0), ("Mix",0.0,1.0,0.0)],
    "Delay": [("Time",0.0,2.0,0.3), ("Repeats",0.0,1.0,0.4), ("Mix",0.0,1.0,0.0)],
    "Juno Chorus": [("Rate",0.1,10.0,0.5), ("Depth",0.0,5.0,1.5), ("Phase",0.0,1.0,0.5), ("Mix",0.0,1.0,0.0)],
    "Distortion": [("Amount",0.0,1.0,0.2), ("Mix",0.0,1.0,0.2), ("",0.0,1.0,0.0)],
    "Compressor": [("Thresh",0.0,1.0,0.1), ("Ratio",1.0,20.0,4.0), ("Gain",0.0,1.0,0.05), ("Att",0.001,0.1,0.01), ("Rel",0.01,1.0,0.1)],
    "Resonant LPF": [("Cutoff",20.0,20000.0,20000.0), ("Res",0.0,1.0,0.0), ("",0.0,1.0,0.0)]
}

# --- CUSTOM WIDGETS ---
class ToolTip:
    def __init__(self, widget, text, app_ref):
        self.widget = widget
        self.text = text
        self.app = app_ref
        self.tw = None
        self._id = None
        self.widget.bind("<Enter>", self.schedule, add="+")
        self.widget.bind("<Leave>", self.unschedule, add="+")
        self.widget.bind("<ButtonPress>", self.unschedule, add="+")

    def schedule(self, event=None):
        if not self.app.enable_hover_tooltips.get(): return
        self.unschedule()
        self._id = self.widget.after(400, self.show)

    def unschedule(self, event=None):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        self.hide()

    def show(self, event=None):
        if self.tw: return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_attributes("-topmost", True)
        self.tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tw, text=self.text, background="#2A2A2A", foreground="#FFFFFF", 
                       relief="solid", borderwidth=1, font=("Trebuchet MS", 9), justify="left")
        lbl.pack(ipadx=4, ipady=2)

    def hide(self):
        if self.tw:
            self.tw.destroy()
            self.tw = None

class CTkKnob(ctk.CTkFrame):
    def __init__(self, master, width=40, height=40, from_=0.0, to=1.0, command=None, fg_color="transparent", progress_color="#3B8ED0", **kwargs):
        super().__init__(master, width=width, height=height, fg_color=fg_color, **kwargs)
        self.min_val, self.max_val, self.command, self.value = from_, to, command, from_
        self.progress_color = progress_color
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
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=225, extent=-270*pct, style="arc", outline=self.progress_color, width=4)
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
    def __init__(self, master, app_ref, default=75, **kwargs):
        super().__init__(master, fg_color="#1F1F1F", corner_radius=6, border_width=1, border_color="#333333", height=65, **kwargs)
        self.app = app_ref
        self.lbl_title = ctk.CTkLabel(self, text="BPM", font=("Trebuchet MS", 9, "bold"), text_color="#2ECC71", height=15)
        self.lbl_title.pack(padx=8, pady=(8, 0))
        self.lbl_val = ctk.CTkLabel(self, text=f"{int(default)}", font=("Trebuchet MS", 15, "bold"), text_color="#2ECC71", height=20)
        self.lbl_val.pack(padx=8, pady=(0, 8))
        
        for w in (self, self.lbl_title, self.lbl_val):
            w.bind("<B1-Motion>", self._on_drag)
            w.bind("<ButtonPress-1>", lambda e: setattr(self, '_last_y', e.y_root))
            w.bind("<Double-Button-1>", self._on_double)
            w.bind("<MouseWheel>", self._on_wheel)
            w.bind("<Button-4>", self._on_wheel)
            w.bind("<Button-5>", self._on_wheel)
        self._last_y = 0
        
    def set_val(self, val):
        self.app.engine.bpm = max(40, min(300, int(val)))
        self.lbl_val.configure(text=f"{int(self.app.engine.bpm)}")
        
    def _on_drag(self, e):
        dy = self._last_y - e.y_root
        if abs(dy) > 2:
            self.app.save_state(); self.set_val(self.app.engine.bpm + (dy // 3)); self._last_y = e.y_root
            
    def _on_wheel(self, e):
        dy = 1 if e.num == 4 or getattr(e, 'delta', 0) > 0 else -1
        self.app.save_state()
        self.set_val(self.app.engine.bpm + dy)

    def _on_double(self, e):
        res = ctk.CTkInputDialog(text="Enter BPM (40-300):", title="BPM Entry").get_input()
        if res:
            try: self.app.save_state(); self.set_val(int(res))
            except ValueError: pass

# --- DSP ENGINE & LFO ---
class NumpyJunoChorus:
    def __init__(self):
        self.history = np.zeros((8192, 2), dtype=np.float32)
        self.phase_l = 0.0
        
    def process(self, audio, sr, rate_hz, depth_ms, phase_offset, mix, bpm, bpm_sync):
        frames = audio.shape[0]
        if frames == 0: return audio
        
        if bpm_sync and bpm > 0:
            bps = bpm / 60.0
            mults = [8, 6, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.25, 0.125]
            available_hz = [bps * m for m in mults]
            rate_hz = min(available_hz, key=lambda x: abs(x - rate_hz))
            
        buf = np.concatenate((self.history, audio), axis=0)
        self.history = buf[-8192:]
        
        t = np.arange(frames) / sr
        
        ph_l = (self.phase_l + t * rate_hz) % 1.0
        ph_r = (self.phase_l + t * rate_hz + phase_offset) % 1.0
        self.phase_l = (self.phase_l + frames * rate_hz / sr) % 1.0
        
        lfo_l = 2.0 * np.abs(2.0 * ph_l - 1.0) - 1.0
        lfo_r = 2.0 * np.abs(2.0 * ph_r - 1.0) - 1.0
        
        depth_samples = depth_ms * 0.001 * sr
        base_delay = depth_samples + 10 
        d_samples_l = base_delay + lfo_l * depth_samples
        d_samples_r = base_delay + lfo_r * depth_samples
        
        idx_l = 8192 + np.arange(frames) - d_samples_l
        idx_r = 8192 + np.arange(frames) - d_samples_r
        
        idx_l = np.clip(idx_l, 0, 8192 + frames - 2)
        idx_r = np.clip(idx_r, 0, 8192 + frames - 2)
        
        idx_l_i = idx_l.astype(int)
        idx_l_f = idx_l - idx_l_i
        
        idx_r_i = idx_r.astype(int)
        idx_r_f = idx_r - idx_r_i
        
        wet = np.zeros_like(audio)
        wet[:, 0] = buf[idx_l_i, 0] * (1 - idx_l_f) + buf[idx_l_i + 1, 0] * idx_l_f
        wet[:, 1] = buf[idx_r_i, 1] * (1 - idx_r_f) + buf[idx_r_i + 1, 1] * idx_r_f
        
        return audio * (1.0 - mix) + wet * mix

class LFO:
    def __init__(self, color):
        self.color, self.shape, self.sync = color, "Sine", True
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
        
        if self.target_id:
            if self.target_id.startswith("gl_rand:") or self.target_id.endswith(":cyc") or self.target_id.endswith(":trigger") or self.target_id.endswith(":roll_euc"):
                if prev_val <= 0 and self.val > 0:
                    trigger_queue.append(self.target_id)
            elif self.target_id == "gl_stutter:gate":
                if prev_val <= 0 and self.val > 0:
                    trigger_queue.append("gl_stutter:on")
                elif prev_val > 0 and self.val <= 0:
                    trigger_queue.append("gl_stutter:off")

class GlobalFXBus:
    def __init__(self, fx_type):
        self.fx_type = fx_type
        self.p1, self.p2, self.p3, self.p4, self.p5 = 0.0, 0.0, 0.0, 0.0, 0.0
        self.m_p1, self.m_p2, self.m_p3, self.m_p4, self.m_p5 = 0.0, 0.0, 0.0, 0.0, 0.0
        self.juno_mode = "MANUAL"
        self.juno_bpm_sync = False
        self.plugin, self.lock = None, threading.Lock(); self.set_type(fx_type)
    def set_type(self, fx_type):
        with self.lock:
            self.fx_type = fx_type
            if fx_type == "Reverb": self.plugin = [Delay(delay_seconds=0.0, feedback=0.0, mix=1.0), Reverb()]
            elif fx_type == "Delay": self.plugin = Delay()
            elif fx_type == "Juno Chorus": self.plugin = NumpyJunoChorus()
            elif fx_type == "Distortion": self.plugin = Distortion()
            elif fx_type == "Compressor": self.plugin = Compressor()
            elif fx_type == "Resonant LPF": self.plugin = LadderFilter(mode=LadderFilter.Mode.LPF12) if HAS_LADDER else LowpassFilter()
            
            defs = FX_DEFS[fx_type]
            for i in range(5):
                val = defs[i][3] if i < len(defs) else 0.0
                setattr(self, f'p{i+1}', val)
                setattr(self, f'm_p{i+1}', val)
            self._apply_params()
    def set_param(self, idx, val, force_manual=True):
        with self.lock:
            setattr(self, f'p{idx}', val)
            if self.fx_type == "Juno Chorus" and idx in [1, 2, 3] and force_manual:
                self.juno_mode = "MANUAL"
    def _apply_params(self):
        if not self.plugin: return
        try:
            if self.fx_type == "Reverb":
                self.plugin[0].delay_seconds = self.m_p3
                self.plugin[1].room_size, self.plugin[1].damping = self.m_p1, 1.0 - self.m_p2
                self.plugin[1].dry_level = 0.0
                self.plugin[1].wet_level = 1.0
            elif self.fx_type == "Delay": self.plugin.delay_seconds, self.plugin.feedback, self.plugin.mix = self.m_p1, self.m_p2, self.m_p3
            elif self.fx_type == "Distortion": self.plugin.drive_db = self.m_p1 * 40.0
            elif self.fx_type == "Compressor": 
                self.plugin.threshold_db, self.plugin.ratio = -(self.m_p1 * 60.0), self.m_p2
                self.plugin.attack_ms = self.m_p4 * 1000.0
                self.plugin.release_ms = self.m_p5 * 1000.0
            elif self.fx_type == "Resonant LPF":
                self.plugin.cutoff_hz = max(20.0, self.m_p1)
                if hasattr(self.plugin, 'resonance'): self.plugin.resonance = self.m_p2
        except Exception: pass
    def process(self, audio_bus, sample_rate, bpm=120):
        with self.lock:
            self._apply_params()
            if self.fx_type in["Distortion", "Resonant LPF"]:
                return audio_bus * (1.0 - (self.m_p2 if self.fx_type == "Distortion" else 1.0)) + self.plugin.process(audio_bus, sample_rate, reset=False) * (self.m_p2 if self.fx_type == "Distortion" else 1.0)
            elif self.fx_type == "Compressor": 
                return self.plugin.process(audio_bus, sample_rate, reset=False) * (10.0 ** ((self.m_p3 * 24.0) / 20.0))
            elif self.fx_type == "Reverb":
                dry = audio_bus * (1.0 - self.m_p4)
                wet = self.plugin[0].process(audio_bus, sample_rate, reset=False)
                wet = self.plugin[1].process(wet, sample_rate, reset=False)
                return dry + (wet * self.m_p4)
            elif self.fx_type == "Juno Chorus":
                return self.plugin.process(audio_bus, sample_rate, self.m_p1, self.m_p2, self.m_p3, self.m_p4, bpm, self.juno_bpm_sync)
            else: 
                return self.plugin.process(audio_bus, sample_rate, reset=False)

class DrumbExtractor:
    def __init__(self): 
        self.last_filepath = None
        self.used_starts = set()

    def extract(self, filepath, method="Focused", is_rescan=False):
        if filepath != self.last_filepath or not is_rescan: 
            self.used_starts.clear()
            self.last_filepath = filepath
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo
        
        if method == "Lazy": res = self._extract_old(y, sr)
        elif method == "Normal": res = self._extract_new(y, sr)
        else: res = self._extract_focused(y, sr)
        
        return res, tempo_val

    def rescan_single_channel(self, filepath, channel_name):
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        
        candidates, new_samples, sl = [],[], int(SAMPLE_RATE * 0.4)
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
                self.used_starts.add(v['start']); new_samples.append(v['audio']); candidates =[c for c in candidates if c['start'] != v['start']]
                
        strict = {"Kick": lambda c: c['cent'] < 1200, "Snare": lambda c: c['zcr'] > 0.12 and 1500 < c['cent'] < 4000, "Closed Hat": lambda c: c['cent'] > 5500, "Open Hat": lambda c: c['cent'] > 4500, "Hi Tom": lambda c: 1000 < c['cent'] < 2000, "Low Tom": lambda c: 500 < c['cent'] < 1200, "Crash": lambda c: c['cent'] > 3000, "Ride": lambda c: c['cent'] > 3000}
        relax = {"Kick": lambda c: c['cent'] < 2000, "Snare": lambda c: c['zcr'] > 0.07, "Closed Hat": lambda c: c['cent'] > 3500, "Open Hat": lambda c: c['cent'] > 3000, "Hi Tom": lambda c: 800 < c['cent'] < 2500, "Low Tom": lambda c: 300 < c['cent'] < 1500, "Crash": lambda c: c['cent'] > 2000, "Ride": lambda c: c['cent'] > 2000}
        if channel_name in strict: add_if(strict[channel_name])
        if channel_name in relax: add_if(relax[channel_name])
        while len(new_samples) < 15:
            if candidates:
                c = random.choice(candidates); self.used_starts.add(c['start']); new_samples.append(c['audio']); candidates =[cand for cand in candidates if cand['start'] != c['start']]
            else: new_samples.append(np.zeros(sl, dtype=np.float32))
        return new_samples

    def _extract_focused(self, y, sr):
        candidates, extracted, sl = [], {c:[] for c in CHANNELS}, int(SAMPLE_RATE * 0.4)
        onsets = librosa.frames_to_samples(librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, wait=1, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.05))
        np.random.shuffle(onsets)
        
        for start in onsets[:800]: 
            if start in self.used_starts: continue
            snippet = y[start:start+sl]
            if len(snippet) < 1024 or np.mean(librosa.feature.rms(y=snippet)) < 0.005: continue
            snippet = snippet / np.max(np.abs(snippet)) if np.max(np.abs(snippet)) > 0 else snippet
            
            cent = np.mean(librosa.feature.spectral_centroid(y=snippet, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=snippet))
            
            S = np.abs(librosa.stft(snippet, n_fft=min(512, len(snippet))))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=min(512, len(snippet)))
            peak_f = freqs[np.argmax(np.mean(S, axis=1))] if S.size > 0 else 0
            
            candidates.append({'audio': snippet, 'cent': cent, 'zcr': zcr, 'peak_f': peak_f, 'start': start})
            
        def run_pass(ch, filt):
            nonlocal candidates
            if (needs := 15 - len(extracted[ch])) <= 0: return
            valid = [c for c in candidates if filt(c)]
            for v in valid[:needs]:
                self.used_starts.add(v['start']); extracted[ch].append(v['audio'])
                candidates = [c for c in candidates if c['start'] != v['start']]
                
        run_pass("Kick", lambda c: c['peak_f'] < 140 or c['cent'] < 300)
        run_pass("Snare", lambda c: c['zcr'] > 0.08 and 1000 < c['cent'] < 3000)
        run_pass("Closed Hat", lambda c: c['cent'] > 5000 and c['zcr'] > 0.15)
        run_pass("Open Hat", lambda c: c['cent'] > 4000)
        run_pass("Hi Tom", lambda c: 600 < c['peak_f'] < 1000 or 600 < c['cent'] < 1200)
        run_pass("Low Tom", lambda c: 300 < c['peak_f'] < 600 or 300 < c['cent'] < 600)
        run_pass("Crash", lambda c: c['cent'] > 3000)
        run_pass("Ride", lambda c: c['cent'] > 3000)
        
        # Fallback
        for ch in extracted:
            while len(extracted[ch]) < 15:
                if candidates:
                    c = random.choice(candidates)
                    self.used_starts.add(c['start']); extracted[ch].append(c['audio'])
                    candidates = [cand for cand in candidates if cand['start'] != c['start']]
                else: extracted[ch].append(np.zeros(sl, dtype=np.float32))
                
        return extracted

    def _extract_new(self, y, sr):
        candidates, extracted, sl =[], {c:[] for c in CHANNELS}, int(SAMPLE_RATE * 0.4) 
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
            
        for m in[{"Kick": lambda c: c['cent'] < 1200, "Snare": lambda c: c['zcr'] > 0.12 and 1500 < c['cent'] < 4000, "Closed Hat": lambda c: c['cent'] > 5500, "Open Hat": lambda c: c['cent'] > 4500, "Hi Tom": lambda c: 1000 < c['cent'] < 2000, "Low Tom": lambda c: 500 < c['cent'] < 1200, "Crash": lambda c: c['cent'] > 3000, "Ride": lambda c: c['cent'] > 3000},
                  {"Kick": lambda c: c['cent'] < 2000, "Snare": lambda c: c['zcr'] > 0.07, "Closed Hat": lambda c: c['cent'] > 3500, "Open Hat": lambda c: c['cent'] > 3000, "Hi Tom": lambda c: 800 < c['cent'] < 2500, "Low Tom": lambda c: 300 < c['cent'] < 1500, "Crash": lambda c: c['cent'] > 2000, "Ride": lambda c: c['cent'] > 2000}]:
            for ch, rule in m.items(): run_pass(ch, rule)
            
        for ch in extracted:
            while len(extracted[ch]) < 15:
                if candidates: c = random.choice(candidates); self.used_starts.add(c['start']); extracted[ch].append(c['audio']); candidates =[cand for cand in candidates if cand['start'] != c['start']]
                else: extracted[ch].append(np.zeros(sl, dtype=np.float32))
                
        return extracted

    def _extract_old(self, y, sr):
        drumbs, onset_samples =[], librosa.frames_to_samples(librosa.onset.onset_detect(y=y, sr=sr, backtrack=True))
        for i, start in enumerate(onset_samples):
            if start in self.used_starts: continue
            snippet = y[start:min(onset_samples[i+1] if i+1 < len(onset_samples) else len(y), start + int(SAMPLE_RATE * 0.4))]
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
            "Ride": mark_used(s_dur[15:30] if len(s_dur) >= 30 else s_dur[:15])
        }
        for k in extracted:
            while len(extracted[k]) < 15: extracted[k].append(extracted[k][0] if extracted[k] else np.zeros(100))
        return extracted

class DrumChannel:
    def __init__(self, name, engine_ref):
        self.name, self.engine, self.samples = name, engine_ref,[np.zeros(1024, dtype=np.float32)] * 15
        self.current_sample_idx = 0; self.m_current_sample_idx = 0
        self.mute, self.solo, self.solo_locked, self.locked, self.trigger_flag = False, False, False, False, False
        self.reverse = False; self.m_reverse = False
        self.pattern_settings =[]
        for _ in range(4): self.pattern_settings.append({'vol':0.8, 'pan':0.0, 'pitch':0, 's_start':0, 's_end':300, 'fx_sends':[0.0, 0.0, 0.0], 'euclid_k':0, 'fade_in':1.0, 'fade_out':100.0, 'smpl_idx':0})
        self.vol, self.pan, self.pitch, self.sample_start, self.sample_end, self.fade_in, self.fade_out, self.euclid_k = 0.8, 0.0, 0, 0, 300, 1.0, 100.0, 0
        self.fx_sends = [0.0, 0.0, 0.0]
        self.m_vol, self.m_pan, self.m_pitch, self.m_sample_start, self.m_sample_end, self.m_fade_in, self.m_fade_out, self.m_euclid_k = 0.8, 0.0, 0, 0, 300, 1.0, 100.0, 0
        self.m_fx_sends = [0.0, 0.0, 0.0]
        self.m_mute, self.m_solo = False, False
        self.sequence = [[False]*32 for _ in range(4)]; self.m_sequence = [[False]*32 for _ in range(4)]
        
    def save_pattern_state(self, pat_idx):
        self.pattern_settings[pat_idx] = {'vol': self.vol, 'pan': self.pan, 'pitch': self.pitch, 's_start': self.sample_start, 's_end': self.sample_end, 'fx_sends': self.fx_sends[:], 'euclid_k': self.euclid_k, 'fade_in': self.fade_in, 'fade_out': self.fade_out, 'smpl_idx': self.current_sample_idx}
        
    def load_pattern_state(self, pat_idx):
        s = self.pattern_settings[pat_idx]
        self.vol, self.pan, self.pitch, self.sample_start, self.sample_end = s['vol'], s['pan'], s['pitch'], s['s_start'], s['s_end']
        self.fx_sends = s.get('fx_sends', [s.get('fx1',0.0), s.get('fx2',0.0), s.get('fx3',0.0)])
        while len(self.fx_sends) < len(self.engine.fx_buses): self.fx_sends.append(0.0)
        self.euclid_k = s['euclid_k']
        self.fade_in, self.fade_out = s.get('fade_in', 1.0), s.get('fade_out', 100.0)
        self.current_sample_idx = s.get('smpl_idx', self.current_sample_idx)

    def get_state(self):
        return {'idx': self.current_sample_idx, 'mute': self.mute, 'solo_l': self.solo_locked, 'locked': self.locked, 'rev': self.reverse, 'ps': copy.deepcopy(self.pattern_settings), 'seq':[r[:] for r in self.sequence]}
        
    def apply_state(self, s, pat):
        self.current_sample_idx, self.mute, self.solo_locked, self.locked = s['idx'], s['mute'], s['solo_l'], s.get('locked', False)
        self.reverse = s.get('rev', False)
        self.pattern_settings = copy.deepcopy(s['ps'])
        self.solo, self.sequence = self.solo_locked, [r[:] for r in s['seq']]
        self.load_pattern_state(pat)

    def step_sample(self, forward=True):
        self.current_sample_idx = (self.current_sample_idx + (1 if forward else -1)) % 15

class AudioEngine:
    def __init__(self):
        self.fx_buses =[GlobalFXBus("Reverb"), GlobalFXBus("Resonant LPF"), GlobalFXBus("Compressor")]
        self.channels =[DrumChannel(name, self) for name in CHANNELS]
        self.lfos =[LFO("#00FFFF"), LFO("#FF00FF"), LFO("#FFFF00"), LFO("#00FF00")]
        self.active_voices, self.record_buffer, self.is_playing, self.is_recording =[],[], False, False
        self.lfo_trigger_queue =[]
        self.bpm = random.randint(55, 85)
        self.steps, self.current_pattern, self.swing = 16, 0, 10
        self.global_vol, self.global_pitch = 0.8, 0
        self.master_fx_sends = [0.0, 0.0, 0.0]
        self.m_global_vol, self.m_global_pitch, self.m_swing = 0.8, 0, 10
        self.m_master_fx_sends = [0.0, 0.0, 0.0]
        self.samples_until_next_step, self.step_counter = 0.0, 0
        self.current_step = 0
        
        self.stutter_active = False
        self.stutter_div_idx = 4
        self.m_stutter_div_idx = 4
        self.stutter_mute_bg = False
        self.stutter_buffer = None
        self.stutter_pos = 0
        self.stutter_recorded = 0
        self.stutter_len = 0
        self.play_bg_during_stutter = True

        self.midi_sync = False
        self.midi_port_name = None
        self.midi_port = None
        self.midi_tick_count = 0
        self.midi_steps_to_trigger = 0
        self.current_device_name = None
        
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=2, callback=self.audio_callback, dtype='float32')
        self.stream.start()

    def set_audio_device(self, dev_idx):
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
            self.current_device_idx = dev_idx
            self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=2, callback=self.audio_callback, dtype='float32', device=dev_idx)
            self.stream.start()
        except Exception as e:
            print(f"Failed to change audio device to index {dev_idx}: {e}")

    def set_midi_device(self, dev_name):
        if not HAS_MIDO: return
        self.midi_port_name = dev_name
        if self.midi_port:
            self.midi_port.close()
            self.midi_port = None
        if dev_name and dev_name != "None":
            try:
                self.midi_port = mido.open_input(dev_name, callback=self.midi_callback)
            except Exception as e:
                print(f"Failed to open MIDI port {dev_name}: {e}")

    def midi_callback(self, msg):
        if not self.midi_sync: return
        now = time.time()
        if msg.type == 'start':
            self.is_playing = True
            self.step_counter = 0
            self.current_step = 0
            self.midi_tick_count = 0
            self.midi_steps_to_trigger = 0
            self.last_midi_clock_time = now
            self.midi_clock_intervals = deque(maxlen=24)
        elif msg.type == 'stop':
            self.is_playing = False
        elif msg.type == 'clock':
            if hasattr(self, 'last_midi_clock_time'):
                delta = now - self.last_midi_clock_time
                if delta < 1.0:
                    if not hasattr(self, 'midi_clock_intervals'): self.midi_clock_intervals = deque(maxlen=24)
                    self.midi_clock_intervals.append(delta)
                    if len(self.midi_clock_intervals) > 6:
                        avg_delta = sum(self.midi_clock_intervals) / len(self.midi_clock_intervals)
                        new_bpm = 60.0 / (avg_delta * 24.0)
                        if 20 <= new_bpm <= 400:
                            self.bpm = new_bpm
                self.last_midi_clock_time = now

            if self.is_playing:
                self.midi_tick_count += 1
                if self.midi_tick_count >= 6:
                    self.midi_tick_count = 0
                    self.midi_steps_to_trigger += 1
        
    def get_state(self):
        return {'bpm': self.bpm, 'steps': self.steps, 'pat': self.current_pattern, 'swing': self.swing, 'gvol': self.global_vol, 'gpitch': self.global_pitch, 'm_fx_sends': self.master_fx_sends[:], 'stutter_div_idx': self.stutter_div_idx, 'ch':[c.get_state() for c in self.channels], 'fx':[{'type': f.fx_type, 'p1': f.p1, 'p2': f.p2, 'p3': f.p3, 'p4': getattr(f, 'p4', 0.0), 'p5': getattr(f, 'p5', 0.0), 'juno_mode': getattr(f, 'juno_mode', 'MANUAL'), 'juno_bpm_sync': getattr(f, 'juno_bpm_sync', False)} for f in self.fx_buses]}
        
    def apply_state(self, s):
        self.bpm, self.steps, self.current_pattern, self.swing, self.global_vol, self.global_pitch = s['bpm'], s['steps'], s['pat'], s['swing'], s['gvol'], s['gpitch']
        self.master_fx_sends = s.get('m_fx_sends', [s.get('m_fx1', 0.0), s.get('m_fx2', 0.0), s.get('m_fx3', 0.0)])
        self.stutter_div_idx = s.get('stutter_div_idx', 4)
        self.m_stutter_div_idx = self.stutter_div_idx
        
        while len(self.fx_buses) < len(s.get('fx',[])):
            self.fx_buses.append(GlobalFXBus("Delay"))
            
        while len(self.master_fx_sends) < len(self.fx_buses):
            self.master_fx_sends.append(0.0)
            
        for i, c in enumerate(self.channels):
            c.apply_state(s['ch'][i], self.current_pattern)
            while len(c.fx_sends) < len(self.fx_buses): c.fx_sends.append(0.0)
            
        for i, fs in enumerate(s['fx']):
            f = self.fx_buses[i]
            if f.fx_type != fs['type']: f.set_type(fs['type'])
            f.set_param(1, fs['p1']); f.set_param(2, fs['p2']); f.set_param(3, fs['p3'])
            f.set_param(4, fs.get('p4', FX_DEFS[fs['type']][3][3] if len(FX_DEFS[fs['type']])>3 else 0.0))
            f.set_param(5, fs.get('p5', FX_DEFS[fs['type']][4][3] if len(FX_DEFS[fs['type']])>4 else 0.0))
            f.juno_mode = fs.get('juno_mode', 'MANUAL')
            f.juno_bpm_sync = fs.get('juno_bpm_sync', False)

    def apply_lfo_target(self, target, val, depth):
        parts, offset = target.split(":"), val * depth
        if parts[0] == "ch":
            ch = self.channels[int(parts[1])]
            param = parts[2]
            if param in['cyc', 'trigger', 'roll_euc']: return
            if param == "fx_sends":
                idx = int(parts[3])
                b = ch.fx_sends[idx]
                ch.m_fx_sends[idx] = max(0.0, min(1.0, b + offset))
            elif param in['mute', 'solo', 'reverse']: setattr(ch, "m_"+param, (float(getattr(ch, param)) + offset) > 0.5)
            else:
                limits = {'vol':(0,1), 'pan':(-1,1), 'pitch':(-24,24), 'sample_start':(0,400), 'sample_end':(0,400), 'euclid_k':(0, self.steps), 'current_sample_idx':(0, 14), 'fade_in':(0, 100), 'fade_out':(0, 100)}
                b, span = getattr(ch, param), limits[param][1] - limits[param][0]
                setattr(ch, "m_"+param, max(limits[param][0], min(limits[param][1], b + offset * span)))
        elif parts[0] == "gl":
            param = parts[1]
            if param == "master_fx":
                idx = int(parts[2])
                b = self.master_fx_sends[idx]
                self.m_master_fx_sends[idx] = max(0.0, min(1.0, b + offset))
            else:
                limits = {'global_vol':(0,1), 'global_pitch':(-24,24), 'swing':(0,100), 'stutter_div_idx':(0,9)}
                b, span = getattr(self, param), limits[param][1] - limits[param][0]
                setattr(self, "m_"+param, max(limits[param][0], min(limits[param][1], b + offset * span)))
        elif parts[0] == "fx":
            fx_b, p_idx = self.fx_buses[int(parts[1])], int(parts[2][1]) - 1
            defs = FX_DEFS[fx_b.fx_type]
            if p_idx < len(defs) and defs[p_idx][0]:
                b, p_min, p_max = getattr(fx_b, parts[2]), defs[p_idx][1], defs[p_idx][2]
                setattr(fx_b, "m_"+parts[2], max(p_min, min(p_max, b + offset * (p_max - p_min))))
        elif parts[0] == "grid":
            ch, step = self.channels[int(parts[1])], int(parts[2])
            ch.m_sequence[self.current_pattern][step] = (float(ch.sequence[self.current_pattern][step]) + offset) > 0.5

    def trigger_channel(self, ch):
        ch.trigger_flag = True; pan = ch.m_pan
        lg, rg = math.cos((pan + 1) * math.pi / 4) * ch.m_vol, math.sin((pan + 1) * math.pi / 4) * ch.m_vol
        idx = int(ch.m_current_sample_idx) % 15
        raw = ch.samples[idx]
        st, en = max(0, min(len(raw)-1, int((ch.m_sample_start / 1000.0) * SAMPLE_RATE))), max(max(0, min(len(raw)-1, int((ch.m_sample_start / 1000.0) * SAMPLE_RATE)))+1, min(len(raw), int((ch.m_sample_end / 1000.0) * SAMPLE_RATE)))
        buf = raw[st:en].copy()
        
        if ch.m_reverse:
            buf = buf[::-1].copy()
            
        buf_len = len(buf)
        if buf_len > 0:
            fi_samples = int((ch.m_fade_in / 100.0) * buf_len)
            fo_samples = int(((100.0 - ch.m_fade_out) / 100.0) * buf_len)
            env = np.ones(buf_len, dtype=np.float32)
            if fi_samples > 0:
                env[:fi_samples] = np.arange(fi_samples, dtype=np.float32) / fi_samples
            if fo_samples > 0:
                env[-fo_samples:] = np.arange(fo_samples, 0, -1, dtype=np.float32) / fo_samples
            elif (fade := min(int(0.005 * SAMPLE_RATE), buf_len)) > 0:
                env[-fade:] = np.arange(fade, 0, -1, dtype=np.float32) / fade
            buf *= env

        self.active_voices.append({'ch': ch, 'buffer': buf, 'pos': 0.0, 'len': len(buf), 'lg': lg, 'rg': rg, 'fx_sends': ch.m_fx_sends[:]})

    def audio_callback(self, outdata, frames, time_info, status):
        num_fx = len(self.fx_buses)
        for ch in self.channels:
            ch.m_vol, ch.m_pan, ch.m_pitch, ch.m_sample_start, ch.m_sample_end = ch.vol, ch.pan, ch.pitch, ch.sample_start, ch.sample_end
            ch.m_euclid_k, ch.m_mute, ch.m_solo = ch.euclid_k, ch.mute, ch.solo
            ch.m_fade_in, ch.m_fade_out = ch.fade_in, ch.fade_out
            ch.m_current_sample_idx = ch.current_sample_idx
            ch.m_reverse = ch.reverse
            for i in range(num_fx):
                ch.m_fx_sends[i] = ch.fx_sends[i]
            for p in range(4):
                for s in range(self.steps): ch.m_sequence[p][s] = ch.sequence[p][s]
        self.m_global_vol, self.m_global_pitch, self.m_swing = self.global_vol, self.global_pitch, self.swing
        self.m_stutter_div_idx = getattr(self, 'stutter_div_idx', 4)
        for i in range(num_fx):
            self.m_master_fx_sends[i] = self.master_fx_sends[i]
            fx = self.fx_buses[i]
            fx.m_p1, fx.m_p2, fx.m_p3 = fx.p1, fx.p2, fx.p3
            fx.m_p4, fx.m_p5 = getattr(fx, 'p4', 0.0), getattr(fx, 'p5', 0.0)

        time_delta = frames / SAMPLE_RATE
        for lfo in self.lfos:
            lfo.step(time_delta, self.bpm, self.lfo_trigger_queue)
            if lfo.target_id and lfo.depth > 0 and not (lfo.target_id.startswith("gl_rand:") or lfo.target_id.endswith(":cyc") or lfo.target_id.endswith(":trigger") or lfo.target_id.endswith(":roll_euc")) and lfo.target_id != "gl_stutter:gate": 
                self.apply_lfo_target(lfo.target_id, lfo.val, lfo.depth)

        for ch in self.channels:
            if ch.m_euclid_k != ch.euclid_k and ch.m_euclid_k > 0:
                k, st = int(ch.m_euclid_k), self.steps
                for i in range(st): ch.m_sequence[self.current_pattern][i] = ((i * k) % st) < k

        out = np.zeros((frames, 2), dtype=np.float32)
        fx_buses_audio = [np.zeros((frames, 2), dtype=np.float32) for _ in range(num_fx)]
        any_soloed = any(ch.m_solo for ch in self.channels)
        
        if self.is_playing:
            if getattr(self, 'midi_sync', False):
                while getattr(self, 'midi_steps_to_trigger', 0) > 0:
                    step_idx = self.step_counter % self.steps
                    self.current_step = step_idx
                    for ch in self.channels:
                        if ch.m_sequence[self.current_pattern][step_idx] and not ch.m_mute:
                            if not any_soloed or ch.m_solo: self.trigger_channel(ch)
                    self.step_counter += 1
                    self.midi_steps_to_trigger -= 1
                self.samples_until_next_step = 0
            else:
                frame_idx = 0
                while frame_idx < frames:
                    if self.samples_until_next_step <= 0:
                        step_idx = self.step_counter % self.steps
                        self.current_step = step_idx
                        for ch in self.channels:
                            if ch.m_sequence[self.current_pattern][step_idx] and not ch.m_mute:
                                if not any_soloed or ch.m_solo: self.trigger_channel(ch)
                        self.step_counter += 1
                        base_sps = (60.0 / max(1.0, float(self.bpm))) * (SAMPLE_RATE / 4.0)
                        sw = base_sps * (float(self.m_swing) / 100.0)
                        self.samples_until_next_step += (base_sps + sw) if self.step_counter % 2 == 1 else (base_sps - sw)
                    
                    advance = min(frames - frame_idx, max(1, int(math.ceil(self.samples_until_next_step))))
                    self.samples_until_next_step -= advance
                    frame_idx += advance

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

                sl_lg, sl_rg = audio_slice * v['lg'], audio_slice * v['rg']
                out[:write_frames, 0] += sl_lg; out[:write_frames, 1] += sl_rg
                
                for f_idx in range(num_fx):
                    m_send = v['fx_sends'][f_idx] if f_idx < len(v['fx_sends']) else 0.0
                    if m_send > 0:
                        fx_buses_audio[f_idx][:write_frames, 0] += sl_lg * m_send
                        fx_buses_audio[f_idx][:write_frames, 1] += sl_rg * m_send
            
            v['pos'] = end_pos
            if v['pos'] < v['len']: active_next.append(v)
                
        self.active_voices = active_next
        
        final_sig = out
        for f_idx, fx_bus in enumerate(self.fx_buses):
            m_send = self.m_master_fx_sends[f_idx]
            final_sig = (final_sig * (1.0 - m_send)) + fx_bus.process((final_sig * m_send) + fx_buses_audio[f_idx], SAMPLE_RATE, self.bpm)
            
        final_sig *= self.m_global_vol

        if getattr(self, 'stutter_active', False) and getattr(self, 'stutter_buffer', None) is not None:
            frames_left = frames
            out_pos = 0
            stutter_out = np.zeros_like(final_sig)
            
            play_bg = self.play_bg_during_stutter and not getattr(self, 'stutter_mute_bg', False)
            
            while frames_left > 0:
                if self.stutter_recorded < self.stutter_len:
                    record_chunk = min(frames_left, self.stutter_len - self.stutter_recorded)
                    fade_len = min(int(0.001 * SAMPLE_RATE), self.stutter_len // 2)
                    
                    chunk_data = final_sig[out_pos : out_pos + record_chunk].copy()
                    idx_arr = np.arange(self.stutter_recorded, self.stutter_recorded + record_chunk, dtype=np.float32)
                    mul = np.ones(record_chunk, dtype=np.float32)
                    
                    fi_mask = idx_arr < fade_len
                    mul[fi_mask] = idx_arr[fi_mask] / fade_len
                    
                    fo_mask = idx_arr >= self.stutter_len - fade_len
                    mul[fo_mask] = (self.stutter_len - 1 - idx_arr[fo_mask]) / fade_len
                    
                    chunk_data *= mul.reshape(-1, 1)
                    
                    self.stutter_buffer[self.stutter_recorded : self.stutter_recorded + record_chunk] = chunk_data
                    if not play_bg:
                        stutter_out[out_pos : out_pos + record_chunk] = chunk_data
                    
                    self.stutter_recorded += record_chunk
                    out_pos += record_chunk
                    frames_left -= record_chunk
                else:
                    play_chunk = min(frames_left, self.stutter_len - self.stutter_pos)
                    stutter_out[out_pos : out_pos + play_chunk] = self.stutter_buffer[self.stutter_pos : self.stutter_pos + play_chunk]
                    self.stutter_pos += play_chunk
                    if self.stutter_pos >= self.stutter_len:
                        self.stutter_pos = 0
                        
                    out_pos += play_chunk
                    frames_left -= play_chunk
                    
            if play_bg:
                final_sig += stutter_out
            else:
                final_sig = stutter_out

        outdata[:] = final_sig
        if self.is_recording: self.record_buffer.append(outdata.copy())


class DrumMachineApp(ctk.CTk):
    def __init__(self, engine):
        super().__init__()
        self.engine, self.extractor, self.current_audio_file, self.first_load_done = engine, DrumbExtractor(), None, False
        self.title("Drumber v10.3"); self.geometry("1450x1075"); ctk.set_appearance_mode("Dark"); self.configure(fg_color="#0A0A0A") 
        self.ch_ui_refs, self.fx_ui_refs, self.lfo_ui_refs, self.g_ui_refs, self.g_lbl_refs = {},[],[], {}, {}
        self.widget_to_step, self.grid_buttons, self.widget_to_lfo_target = {}, {}, {}
        self.target_updater_map = {}
        self.grid_bgs = {}
        self.last_played_step = -1
        self.options_window = None
        
        self.auto_detect_bpm = ctk.BooleanVar(value=False)
        self.scan_method_var = ctk.StringVar(value="Focused")
        self.show_tips_var = ctk.BooleanVar(value=True)
        self.enable_hover_tooltips = ctk.BooleanVar(value=True)
        
        self.tips_pool =[
            "You can always load a new track even during playback!",
            "Double click a fader to reset it to default values!",
            "If you choose a new pattern and it is blank, it will copy your existing pattern!",
            "Don't forget that global pitch knob!",
            "Right click a channel name to send the current sample to another track!",
            "Click and hold a channel name to load a new set of samples just for that track!"
        ]
        self.tip_queue =[]
        self.recent_tips = deque(maxlen=8)
        self.last_tip_time = 0
        
        self.undo_stack = deque(maxlen=24)
        self.rand_history = {k: deque(maxlen=8) for k in['Cyc', 'Euc', 'Sam', 'Pan', 'Fad', 'ALL']}
        self.mute_history, self.solo_history, self.lock_history, self.rev_history = deque(maxlen=8), deque(maxlen=8), deque(maxlen=8), deque(maxlen=8)
        self.ctrl_pressed = False
        
        self.bind_all("<B1-Motion>", lambda e: self.on_paint(e, True)); self.bind_all("<B3-Motion>", lambda e: self.on_paint(e, False))
        self.bind_all("<ButtonPress-1>", self.on_global_click_save_state, add="+"); self.bind_all("<ButtonPress-3>", lambda e: self.on_paint(e, False), add="+")
        self.bind_all("<KeyPress-Control_L>", lambda e: setattr(self, 'ctrl_pressed', True)); self.bind_all("<KeyRelease-Control_L>", lambda e: setattr(self, 'ctrl_pressed', False))
        self.bind_all("<KeyPress-Control_R>", lambda e: setattr(self, 'ctrl_pressed', True)); self.bind_all("<KeyRelease-Control_R>", lambda e: setattr(self, 'ctrl_pressed', False))
        self.bind_all("<Button-2>", self.show_lfo_menu)
        self.bind_all("<space>", self.on_spacebar)

        self.build_ui(); self.save_state(); self.rand_pan(); self.gui_update_loop()
        self.after(500, self.initial_auto_load)
        self.after(1000, self.tips_loop)
        
    def show_tip(self, text):
        if text not in self.tip_queue and text not in self.recent_tips:
            self.tip_queue.append(text)

    def force_new_tip(self, e=None):
        if self.show_tips_var.get() and hasattr(self, 'lbl_tips'):
            valid_tips = [t for t in self.tips_pool if t not in self.recent_tips]
            if not valid_tips:
                self.recent_tips.clear()
                valid_tips = self.tips_pool
            if valid_tips:
                text = random.choice(valid_tips)
                self.lbl_tips.configure(text=text)
                self.recent_tips.append(text)
                self.last_tip_time = time.time()

    def tips_loop(self):
        now = time.time()
        if now - self.last_tip_time >= 15.0:
            if self.tip_queue:
                text = self.tip_queue.pop(0)
                if self.show_tips_var.get() and hasattr(self, 'lbl_tips'):
                    self.lbl_tips.configure(text=text)
                self.recent_tips.append(text)
                self.last_tip_time = now
            else:
                valid_tips = [t for t in self.tips_pool if t not in self.recent_tips]
                if not valid_tips:
                    self.recent_tips.clear()
                    valid_tips = self.tips_pool
                if valid_tips:
                    text = random.choice(valid_tips)
                    if self.show_tips_var.get() and hasattr(self, 'lbl_tips'):
                        self.lbl_tips.configure(text=text)
                    self.recent_tips.append(text)
                    self.last_tip_time = now
        self.after(1000, self.tips_loop)

    def initial_auto_load(self):
        self.load_file()

    def gui_update_loop(self):
        while self.engine.lfo_trigger_queue:
            t_id = self.engine.lfo_trigger_queue.pop(0)
            if t_id.startswith("gl_rand:"): self.do_global_rand(t_id.split(":")[1])
            elif t_id.startswith("ch:") and t_id.endswith(":cyc"): self.cycle_drumb(self.engine.channels[int(t_id.split(":")[1])])
            elif t_id.startswith("ch:") and t_id.endswith(":trigger"): self.engine.trigger_channel(self.engine.channels[int(t_id.split(":")[1])])
            elif t_id.startswith("ch:") and t_id.endswith(":roll_euc"): self.roll_single_euc(self.engine.channels[int(t_id.split(":")[1])])
            elif t_id == "gl_stutter:on": self._on_stutter_press(None, mute_bg=False)
            elif t_id == "gl_stutter:off": self._on_stutter_release(None)
            
        for i, ch in enumerate(self.engine.channels):
            if ch.trigger_flag:
                ch.trigger_flag = False; led = self.ch_ui_refs[ch.name]['led']; led.configure(fg_color=ROW_COLORS[i])
                self.after(100, lambda l=led: l.configure(fg_color="#222222"))
                
            m_col = "#CC0000" if ch.m_mute else "#444444"
            btn_mute = self.ch_ui_refs[ch.name]['btn_mute']
            if btn_mute.cget("fg_color") != m_col: btn_mute.configure(fg_color=m_col)
                
            s_col = "#CCCC00" if ch.solo else "#444444"
            btn_solo = self.ch_ui_refs[ch.name]['btn_solo']
            if btn_solo.cget("fg_color") != s_col: btn_solo.configure(fg_color=s_col)
            
            r_col = "#9B59B6" if ch.m_reverse else "#444444"
            btn_rev = self.ch_ui_refs[ch.name]['btn_rev']
            if btn_rev.cget("fg_color") != r_col: btn_rev.configure(fg_color=r_col)
            
        self.refresh_grid_ui()

        # Handle Playback Location Step Highlight
        new_step = self.engine.current_step if self.engine.is_playing else -1
        if getattr(self, 'last_played_step', -1) != new_step:
            old_step = getattr(self, 'last_played_step', -1)
            
            if old_step != -1 and old_step < self.engine.steps:
                for i in range(len(self.engine.channels)):
                    if (i, old_step) in self.grid_bgs:
                        bg_col = "#151515" if (old_step // 4) % 2 == 0 else "#252525"
                        self.grid_bgs[(i, old_step)].configure(fg_color=bg_col, border_color=bg_col)
                        
            if new_step != -1 and new_step < self.engine.steps:
                for i in range(len(self.engine.channels)):
                    if (i, new_step) in self.grid_bgs:
                        self.grid_bgs[(i, new_step)].configure(fg_color="#3A3A3A", border_color=ROW_COLORS[i])
            
            self.last_played_step = new_step

        for i, lfo in enumerate(self.engine.lfos):
            bright = int(((lfo.val + 1.0) / 2.0) * 255)
            self.lfo_ui_refs[i]['led'].configure(fg_color=f"#{bright:02x}{bright:02x}{bright:02x}")
            if lfo.target_id and not (lfo.target_id.startswith("gl_rand:") or lfo.target_id.endswith(":cyc") or lfo.target_id.endswith(":trigger") or lfo.target_id.endswith(":roll_euc") or lfo.target_id == "gl_stutter:gate") and lfo.target_id in self.target_updater_map:
                try: self.target_updater_map[lfo.target_id]()
                except Exception: pass

        for i, fx_bus in enumerate(self.engine.fx_buses):
            if fx_bus.fx_type == "Juno Chorus" and self.fx_ui_refs[i] and 'juno_btns' in self.fx_ui_refs[i]:
                btns = self.fx_ui_refs[i]['juno_btns']
                for m, btn in btns.items():
                    target_color = "#D35400" if fx_bus.juno_mode == m else "#333"
                    if btn.cget("fg_color") != target_color:
                        btn.configure(fg_color=target_color)
                
                bpm_btn = self.fx_ui_refs[i]['juno_bpm_btn']
                target_bpm_color = "#2ECC71" if fx_bus.juno_bpm_sync else "#333"
                if bpm_btn.cget("fg_color") != target_bpm_color:
                    bpm_btn.configure(fg_color=target_bpm_color)

        if HAS_MIDO and getattr(self.engine, 'midi_sync', False):
            self.bpm_widget.set_val(self.engine.bpm)
                
        self.after(30, self.gui_update_loop)

    def on_spacebar(self, e):
        focused = self.focus_get()
        if focused and ("entry" in focused.winfo_class().lower() or "dialog" in focused.winfo_class().lower()):
            return
        self.toggle_play()
        
    def save_state(self): self.undo_stack.append(self.engine.get_state())
    def undo(self):
        if len(self.undo_stack) > 1:
            self.undo_stack.pop(); self.engine.apply_state(self.undo_stack[-1]); self.sync_ui_to_engine(); self.lbl_status.configure(text="Global undo successful.")

    def _save_rand_undo(self, key):
        attrs = ['current_sample_idx'] if key=='Cyc' else ['euclid_k', 'sequence'] if key=='Euc' else['sample_start', 'sample_end'] if key=='Sam' else ['pan'] if key=='Pan' else['fade_in', 'fade_out'] if key=='Fad' else['current_sample_idx', 'euclid_k', 'sequence', 'sample_start', 'sample_end', 'pan', 'fade_in', 'fade_out']
        snap = {}
        for ch in self.engine.channels:
            if not ch.locked: snap[ch.name] = {a: [r[:] for r in ch.sequence] if a == 'sequence' else getattr(ch, a) for a in attrs}
        self.rand_history[key].append(snap)

    def _apply_rand_undo(self, key):
        self.show_tip("Right click a global button to 'undo' previous action!")
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

    def _update_lfo_target_label(self, idx, text):
        font_size = 12
        if len(text) > 15: font_size = max(8, 12 - (len(text) - 15) // 2)
        self.lfo_ui_refs[idx]['lbl_target'].configure(text=text, font=("Trebuchet MS", font_size, "bold"))

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
            self.show_tip("Middle-click any control to send to LFO!")
            menu = tk.Menu(self, tearoff=0, bg="#222", fg="white", font=APP_FONT)
            for i in range(len(self.engine.lfos)):
                menu.add_command(label=f"Assign '{t_name}' to LFO {i+1}", command=lambda idx=i: self.assign_lfo(idx, t_id, t_name))
            menu.tk_popup(e.x_root, e.y_root)

    def show_channel_swap_menu(self, e, src_ch):
        menu = tk.Menu(self, tearoff=0, bg="#222", fg="white", font=APP_FONT)
        for tgt_ch in self.engine.channels:
            if tgt_ch != src_ch:
                menu.add_command(label=f"Send sample to: {tgt_ch.name}", command=lambda s=src_ch, t=tgt_ch: self.swap_samples(s, t))
        menu.tk_popup(e.x_root, e.y_root)

    def swap_samples(self, src_ch, tgt_ch):
        self.save_state()
        src_idx = int(src_ch.current_sample_idx)
        tgt_idx = random.randint(0, 14)
        
        src_sample = src_ch.samples[src_idx].copy()
        tgt_sample = tgt_ch.samples[tgt_idx].copy()
        
        src_ch.samples[src_idx] = tgt_sample
        tgt_ch.samples[tgt_idx] = src_sample
        
        tgt_ch.current_sample_idx = tgt_idx
        if tgt_ch.name in self.ch_ui_refs:
            self.ch_ui_refs[tgt_ch.name]['current_sample_idx'][0].set(tgt_idx)
            self.ch_ui_refs[tgt_ch.name]['current_sample_idx'][1].configure(text=f"{tgt_idx}")
            
        self.lbl_status.configure(text=f"Swapped {src_ch.name} drumb {src_idx+1} with {tgt_ch.name} drumb {tgt_idx+1}")

    def assign_lfo(self, idx, t_id, t_name):
        self.engine.lfos[idx].target_id = t_id
        self.engine.lfos[idx].target_name = t_name
        self._update_lfo_target_label(idx, t_name)
        self.lfo_ui_refs[idx]['btn_clear'].pack(side="right")
        
    def do_global_rand(self, key):
        self.show_tip("Hold a global randomize button to reset it to default values!")
        self.save_state(); self._save_rand_undo(key)
        if key == 'Cyc': self.rand_cycl()
        elif key == 'Euc': self.rand_euc()
        elif key == 'Sam': self.rand_sam()
        elif key == 'Pan': self.rand_pan()
        elif key == 'Fad': self.rand_fade()
        elif key == 'ALL': self.rand_all()
        
    def reset_rand(self, key):
        self.save_state()
        self.show_tip("Hold a global randomize button to reset it to default values!")
        for ch in self.engine.channels:
            if ch.locked: continue
            if key in ('Cyc', 'ALL'): self.handle_ch_slider(ch, 'current_sample_idx', 0)
            if key in ('Euc', 'ALL'): self.handle_ch_slider(ch, 'euclid_k', 0)
            if key in ('Sam', 'ALL'): self.handle_ch_slider(ch, 'sample', 0, True, 400)
            if key in ('Fad', 'ALL'): self.handle_ch_slider(ch, 'fade', 1.0, True, 100.0)
            if key in ('Pan', 'ALL'): self.handle_ch_slider(ch, 'pan', 0.0)

    def _on_stutter_press(self, e, mute_bg=False):
        self.show_tip("Right click stutter to mute background audio during stuttering!")
        self.engine.stutter_active = False 
        val = getattr(self.engine, 'm_stutter_div_idx', getattr(self.engine, 'stutter_div_idx', 4))
        div_idx = max(0, min(len(STUTTER_OPTS)-1, int(round(val))))
        div = STUTTER_OPTS[div_idx][0]
        self.engine.stutter_divisions = div
        self.engine.stutter_mute_bg = mute_bg
        self.engine.stutter_len = max(1, int((60.0 / max(1.0, float(self.engine.bpm)) * 4.0) / div * SAMPLE_RATE))
        self.engine.stutter_buffer = np.zeros((self.engine.stutter_len, 2), dtype=np.float32)
        self.engine.stutter_pos = 0
        self.engine.stutter_recorded = 0
        self.engine.stutter_active = True
        if hasattr(self, 'btn_stutter'):
            self.btn_stutter.configure(fg_color="#8E44AD" if mute_bg else "#D35400")

    def _on_stutter_release(self, e):
        self.engine.stutter_active = False
        if hasattr(self, 'btn_stutter'):
            self.btn_stutter.configure(fg_color="#E67E22")

    def animate_scan(self, step=0):
        if not getattr(self, 'is_scanning', False):
            self.btn_load_scan.place(x=2.5, y=2)
            self.btn_load_scan.configure(fg_color=["#3a7ebf", "#1f538d"])
            return
        dx = [0.5, 4.5, 0.5, 4.5][step % 4]
        self.btn_load_scan.place(x=dx, y=2)
        col = "#E67E22" if (step // 2) % 2 == 0 else "#2980B9"
        self.btn_load_scan.configure(fg_color=col)
        self.after(50, lambda: self.animate_scan(step + 1))

    def draw_global_texture(self, event=None):
        if not hasattr(self, 'global_bg_canvas'): return
        self.global_bg_canvas.delete("all")
        w, h = self.global_frame.winfo_width(), self.global_frame.winfo_height()
        if w < 10 or h < 10: return
        for y in range(0, h, 4):
            self.global_bg_canvas.create_line(0, y, w, y, fill="#191919")

    def build_ui(self):
        top_frame = ctk.CTkFrame(self, fg_color="#141414"); top_frame.pack(fill="x", padx=10, pady=5)
        
        self.load_btn_frame = ctk.CTkFrame(top_frame, width=105, height=32, fg_color="transparent")
        self.load_btn_frame.pack_propagate(False)
        self.load_btn_frame.pack(side="left", padx=5)
        self.btn_load_scan = ctk.CTkButton(self.load_btn_frame, text="Load & Scan", font=APP_FONT_BOLD, width=100, height=28, command=self.load_file)
        self.btn_load_scan.place(x=2.5, y=2)
        
        self.btn_rescan = ctk.CTkButton(top_frame, text="RESCAN", font=APP_FONT_BOLD, width=80, fg_color="#333333", state="disabled", command=self.rescan_file)
        self.btn_rescan.pack(side="left", padx=5)
        self.lbl_status = ctk.CTkLabel(top_frame, text="Ready. Load an audio file.", font=APP_FONT); self.lbl_status.pack(side="left", padx=10)
        
        self.btn_settings = ctk.CTkButton(top_frame, text="⚙", width=30, font=APP_FONT, fg_color="transparent", border_width=1, command=self.open_options)
        self.btn_settings.pack(side="right", padx=5)
        self.lbl_tips = ctk.CTkLabel(top_frame, text="", font=("Trebuchet MS", 11, "italic"), text_color="#777777", cursor="hand2")
        self.lbl_tips.pack(side="right", padx=10)
        self.lbl_tips.bind("<Button-1>", self.force_new_tip)
        
        # Global Frame (Textured Background)
        self.global_frame = ctk.CTkFrame(self, fg_color="#141414")
        self.global_frame.pack(fill="x", padx=10, pady=5)
        self.global_bg_canvas = ctk.CTkCanvas(self.global_frame, bg="#141414", highlightthickness=0)
        self.global_bg_canvas.place(relwidth=1, relheight=1)
        self.global_frame.bind("<Configure>", self.draw_global_texture)

        global_controls = ctk.CTkFrame(self.global_frame, fg_color="transparent"); global_controls.pack(fill="x", padx=5, pady=6)

        # Global Vol Knob
        self.add_global_knob(global_controls, "Vol", 0, 1, 0.8, lambda v: setattr(self.engine, 'global_vol', v), "gl:global_vol")

        # Play / Record Buttons
        play_rec_f = ctk.CTkFrame(global_controls, fg_color="transparent")
        play_rec_f.pack(side="left", padx=5, pady=0)
        self.btn_play = ctk.CTkButton(play_rec_f, text="▶", font=("Arial", 38), width=65, height=65, command=self.toggle_play, fg_color="#333333", state="disabled")
        self.btn_play.pack(side="left", padx=4)
        self.btn_rec = ctk.CTkButton(play_rec_f, text="RECORD", font=("Arial", 6, "bold"), width=32, height=32, corner_radius=16, command=self.toggle_record, fg_color="#333333", text_color="#FFFFFF", state="disabled")
        self.btn_rec.pack(side="left", padx=4)

        # Pitch & Swing
        self.add_global_knob(global_controls, "Pitch", -24, 24, 0, lambda v: setattr(self.engine, 'global_pitch', v), "gl:global_pitch", is_int=True)
        self.add_global_knob(global_controls, "Swing", 0, 100, 10, lambda v: setattr(self.engine, 'swing', v), "gl:swing", is_int=True)
        
        # BPM Widget
        self.bpm_widget = DraggableBPM(global_controls, self, default=75); self.bpm_widget.pack(side="left", padx=2)
        
        # Pattern and Steps
        pat_step_outer = ctk.CTkFrame(global_controls, fg_color="transparent")
        pat_step_outer.pack(side="left", padx=(30, 30))
        pat_frame = ctk.CTkFrame(pat_step_outer, fg_color="transparent")
        pat_frame.pack(side="top", pady=1, fill="x")
        ctk.CTkLabel(pat_frame, text="Pattern", font=APP_FONT).pack(side="left", padx=2)
        self.pat_var = ctk.StringVar(value="0"); ctk.CTkSegmentedButton(pat_frame, values=["0", "1", "2", "3"], font=APP_FONT, variable=self.pat_var, command=self.change_pattern, width=80).pack(side="right")
        step_frame = ctk.CTkFrame(pat_step_outer, fg_color="transparent")
        step_frame.pack(side="top", pady=1, fill="x")
        ctk.CTkLabel(step_frame, text="Steps", font=APP_FONT).pack(side="left", padx=2)
        self.step_var = ctk.StringVar(value="16"); ctk.CTkSegmentedButton(step_frame, values=["8", "16", "32"], font=APP_FONT, variable=self.step_var, command=self.change_steps, width=80).pack(side="right")
        
        # Stutter Controls
        stutter_f = ctk.CTkFrame(global_controls, fg_color="transparent"); stutter_f.pack(side="left", padx=(0, 15))
        self.btn_stutter = ctk.CTkButton(stutter_f, text="STUTTER", width=55, height=65, font=APP_FONT_BOLD, fg_color="#E67E22", hover_color="#D35400")
        self.btn_stutter.pack(side="left", padx=2)
        self.btn_stutter.bind("<ButtonPress-1>", lambda e: self._on_stutter_press(e, False))
        self.btn_stutter.bind("<ButtonPress-3>", lambda e: self._on_stutter_press(e, True))
        self.btn_stutter.bind("<ButtonRelease-1>", self._on_stutter_release)
        self.btn_stutter.bind("<ButtonRelease-3>", self._on_stutter_release)
        self.widget_to_lfo_target[str(self.btn_stutter)] = ("gl_stutter:gate", "Stutter Trigger")
        
        stutter_ctrl_container = ctk.CTkFrame(stutter_f, fg_color="transparent")
        stutter_ctrl_container.pack(side="left", padx=(5,0))

        divs_f = ctk.CTkFrame(stutter_ctrl_container, fg_color="transparent")
        divs_f.pack(side="left")

        r1 = ctk.CTkFrame(divs_f, fg_color="transparent")
        r1.pack(side="top", pady=0)
        r2 = ctk.CTkFrame(divs_f, fg_color="transparent")
        r2.pack(side="top", pady=0)

        def get_current_trip():
            curr = getattr(self.engine, 'stutter_div_idx', 4)
            return 1 if int(round(curr)) % 2 != 0 else 0

        def set_stutter_div(base_idx):
            trip = get_current_trip()
            new_val = base_idx + trip
            self.engine.stutter_div_idx = new_val
            self.engine.m_stutter_div_idx = new_val
            self.save_state()
            if "gl:stutter_div_idx" in self.target_updater_map:
                self.target_updater_map["gl:stutter_div_idx"]()

        def toggle_triplet():
            curr = int(round(getattr(self.engine, 'stutter_div_idx', 4)))
            base = (curr // 2) * 2
            trip = 1 if curr % 2 == 0 else 0
            new_val = base + trip
            self.engine.stutter_div_idx = new_val
            self.engine.m_stutter_div_idx = new_val
            self.save_state()
            if "gl:stutter_div_idx" in self.target_updater_map:
                self.target_updater_map["gl:stutter_div_idx"]()

        btn_1_4 = ctk.CTkButton(r1, text="1/4", width=34, height=32, font=("Trebuchet MS", 9), command=lambda: set_stutter_div(0))
        btn_1_4.pack(side="left", padx=1)
        btn_1_8 = ctk.CTkButton(r1, text="1/8", width=34, height=32, font=("Trebuchet MS", 9), command=lambda: set_stutter_div(2))
        btn_1_8.pack(side="left", padx=1)
        btn_1_16 = ctk.CTkButton(r1, text="1/16", width=34, height=32, font=("Trebuchet MS", 9), command=lambda: set_stutter_div(4))
        btn_1_16.pack(side="left", padx=1)

        btn_1_32 = ctk.CTkButton(r2, text="1/32", width=34, height=32, font=("Trebuchet MS", 9), command=lambda: set_stutter_div(6))
        btn_1_32.pack(side="left", padx=1)
        btn_1_64 = ctk.CTkButton(r2, text="1/64", width=34, height=32, font=("Trebuchet MS", 9), command=lambda: set_stutter_div(8))
        btn_1_64.pack(side="left", padx=1)

        btn_x3 = ctk.CTkButton(stutter_ctrl_container, text="X3", width=20, height=65, font=("Trebuchet MS", 9, "bold"), command=toggle_triplet)
        btn_x3.pack(side="left", padx=(2, 0))

        self.stutter_btns = {0: btn_1_4, 2: btn_1_8, 4: btn_1_16, 6: btn_1_32, 8: btn_1_64}
        self.btn_x3 = btn_x3

        def stutter_updater():
            val = getattr(self.engine, 'm_stutter_div_idx', getattr(self.engine, 'stutter_div_idx', 4))
            idx = max(0, min(9, int(round(val))))
            base = (idx // 2) * 2
            is_trip = (idx % 2 != 0)
            
            for b_idx, btn in self.stutter_btns.items():
                col = "#E67E22" if b_idx == base else "#333333"
                txt_col = "#000000" if b_idx == base else "#FFFFFF"
                if btn.cget("fg_color") != col:
                    btn.configure(fg_color=col)
                if btn.cget("text_color") != txt_col:
                    btn.configure(text_color=txt_col)
                    
            t_col = "#E67E22" if is_trip else "#333333"
            t_txt_col = "#000000" if is_trip else "#FFFFFF"
            if self.btn_x3.cget("fg_color") != t_col:
                self.btn_x3.configure(fg_color=t_col)
            if self.btn_x3.cget("text_color") != t_txt_col:
                self.btn_x3.configure(text_color=t_txt_col)

        self.target_updater_map["gl:stutter_div_idx"] = stutter_updater

        def on_stutter_wheel(e):
            if (dir_step := (1 if e.delta > 0 else -1) if hasattr(e, 'delta') and e.delta != 0 else (1 if e.num == 4 else -1)):
                self.save_state()
                curr = int(round(getattr(self.engine, 'stutter_div_idx', 4)))
                new_val = max(0, min(9, curr + dir_step))
                self.engine.stutter_div_idx = new_val
                self.engine.m_stutter_div_idx = new_val
                if "gl:stutter_div_idx" in self.target_updater_map:
                    self.target_updater_map["gl:stutter_div_idx"]()

        for w in [stutter_ctrl_container, btn_1_4, btn_1_8, btn_1_16, btn_1_32, btn_1_64, btn_x3, divs_f, r1, r2]:
            w.bind("<MouseWheel>", on_stutter_wheel)
            w.bind("<Button-4>", on_stutter_wheel)
            w.bind("<Button-5>", on_stutter_wheel)
            self.widget_to_lfo_target[str(w)] = ("gl:stutter_div_idx", "Stutter Div")
        
        # Global Randomize Array
        rand_outer = ctk.CTkFrame(global_controls, fg_color="transparent")
        rand_outer.pack(side="left", padx=5)
        
        lbl_f_rand = ctk.CTkFrame(rand_outer, fg_color="transparent")
        lbl_f_rand.pack(side="top", pady=(0,2))
        ctk.CTkLabel(lbl_f_rand, text="GLOBAL RANDOMIZE", font=("Trebuchet MS", 10, "bold")).pack(side="left")
        ctk.CTkLabel(lbl_f_rand, text=" (hold button to reset)", font=("Trebuchet MS", 8, "italic"), text_color="#777777").pack(side="left")
        
        rand_frame = ctk.CTkFrame(rand_outer, fg_color="transparent"); rand_frame.pack(side="top")
        
        def mk_rnd(txt, key, tooltip_text, col=None):
            b = ctk.CTkButton(rand_frame, text=txt, width=48, font=APP_FONT)
            if col: b.configure(fg_color=col)
            
            if key == 'ALL': b.pack(side="left", padx=(15, 1))
            else: b.pack(side="left", padx=1)
            
            b.bind("<ButtonPress-1>", lambda e, btn=b: setattr(btn, '_press_time', time.time()))
            b.bind("<ButtonRelease-1>", lambda e, k=key, btn=b: self.reset_rand(k) if (time.time() - getattr(btn, '_press_time', time.time())) >= 0.8 else self.do_global_rand(k))
            b.bind("<Button-3>", lambda e, k=key: self._apply_rand_undo(k))
            self.widget_to_lfo_target[str(b)] = (f"gl_rand:{key}", f"Rand {txt}")
            ToolTip(b, tooltip_text, self)
            
        mk_rnd("Smpl", 'Cyc', "Randomize Sample\nLeft-click to pick a new sample for all unlocked tracks.\nHold to reset.\nRight-click to undo."); 
        mk_rnd("Euc", 'Euc', "Randomize Euclidean\nLeft-click to apply new euclidean rhythms for all unlocked tracks.\nHold to reset.\nRight-click to undo."); 
        mk_rnd("Sam", 'Sam', "Randomize Sample Length\nLeft-click to randomize start/end points.\nHold to reset.\nRight-click to undo."); 
        mk_rnd("Fad", 'Fad', "Randomize Fades\nLeft-click to randomize fade in/out.\nHold to reset.\nRight-click to undo."); 
        mk_rnd("Pan", 'Pan', "Randomize Pan\nLeft-click to randomize panning.\nHold to reset.\nRight-click to undo."); 
        mk_rnd("ALL", 'ALL', "Randomize All Parameters\nLeft-click to randomize everything globally.\nHold to reset.\nRight-click to undo.", "#A93226")
        
        # Global Reset Controls
        reset_outer = ctk.CTkFrame(global_controls, fg_color="transparent")
        reset_outer.pack(side="right", padx=5)
        ctk.CTkLabel(reset_outer, text="GLOBAL CHANNEL OPTION RESET/TOGGLE", font=("Trebuchet MS", 10, "bold")).pack(side="top", pady=(0,2))
        act_f = ctk.CTkFrame(reset_outer, fg_color="transparent"); act_f.pack(side="top")
        
        btn_undo = ctk.CTkButton(act_f, text="UNDO", width=45, font=APP_FONT_BOLD, command=self.undo, fg_color="#1E8449"); btn_undo.pack(side="left", padx=1)
        ToolTip(btn_undo, "Global Undo\nLeft-click to revert the last global action.", self)
        
        btn_unl = ctk.CTkButton(act_f, text="UNLOCK", width=45, font=APP_FONT_BOLD, command=self.unlock_all, fg_color="#8E44AD"); btn_unl.bind("<Button-3>", self.undo_unsolo_all); btn_unl.pack(side="left", padx=1)
        ToolTip(btn_unl, "Unlock All\nLeft-click to unlock all channels.\nRight-click to undo.", self)
        
        btn_uns = ctk.CTkButton(act_f, text="UNSOLO", width=45, font=APP_FONT_BOLD, command=self.unsolo_all, fg_color="#9C640C"); btn_uns.bind("<Button-3>", self.undo_unsolo_all); btn_uns.pack(side="left", padx=1)
        ToolTip(btn_uns, "Unsolo All\nLeft-click to unsolo all channels.\nRight-click to undo.", self)
        
        btn_unm = ctk.CTkButton(act_f, text="UNMUTE", width=45, font=APP_FONT_BOLD, command=self.unmute_all, fg_color="#922B21"); btn_unm.bind("<Button-3>", self.undo_unmute_all); btn_unm.pack(side="left", padx=1)
        ToolTip(btn_unm, "Unmute All\nLeft-click to unmute all channels.\nRight-click to undo.", self)
        
        btn_unrev = ctk.CTkButton(act_f, text="UNREV", width=45, font=APP_FONT_BOLD, command=self.unreverse_all, fg_color="#7D3C98"); btn_unrev.bind("<Button-3>", self.undo_unreverse_all); btn_unrev.pack(side="left", padx=1)
        ToolTip(btn_unrev, "Unreverse All\nLeft-click to un-reverse all channels.\nRight-click to undo.", self)
        
        btn_all = ctk.CTkButton(act_f, text="ALL", width=45, font=APP_FONT_BOLD, command=self.reset_all_params, fg_color="#6B1D1D"); btn_all.pack(side="left", padx=(15,1))
        ToolTip(btn_all, "Reset All\nLeft-click to reset all channels and FX to default settings.", self)

        main_paned = ctk.CTkFrame(self, fg_color="transparent"); main_paned.pack(fill="both", expand=True, padx=10, pady=(5,0))
        top_band = ctk.CTkFrame(main_paned, fg_color="transparent"); top_band.pack(fill="both", expand=True)
        mixer_container = ctk.CTkFrame(top_band, fg_color="transparent"); mixer_container.pack(side="left", fill="both", expand=True) 
        mixer_frame = ctk.CTkFrame(mixer_container, fg_color="#111111"); mixer_frame.pack(pady=2, fill="both", expand=True)
        self.fx_frame = ctk.CTkScrollableFrame(top_band, width=330, fg_color="#141414"); self.fx_frame.pack(side="right", fill="y", padx=(5,0))

        # Right Panel Structure
        self.fx_racks_inner = ctk.CTkFrame(self.fx_frame, fg_color="transparent")
        self.fx_racks_inner.pack(fill="x")
        
        self.btn_add_fx = ctk.CTkButton(self.fx_frame, text="+ Add FX", command=self.add_fx_bus, font=APP_FONT_BOLD)
        self.btn_add_fx.pack(pady=(5, 10))
        
        self.lfo_racks_inner = ctk.CTkFrame(self.fx_frame, fg_color="transparent")
        self.lfo_racks_inner.pack(fill="x", pady=10)

        # Build Mixer Channels
        for i, ch in enumerate(self.engine.channels):
            self.ch_ui_refs[ch.name] = {}
            col = ctk.CTkFrame(mixer_frame, width=130, fg_color="#1A1A1A"); col.pack(side="left", fill="y", expand=True, padx=2, pady=1) 
            top_cf = ctk.CTkFrame(col, fg_color="transparent"); top_cf.pack(pady=6)
            led = ctk.CTkFrame(top_cf, width=8, height=8, corner_radius=4, fg_color="#222222"); led.pack(side="left", padx=2); self.ch_ui_refs[ch.name]['led'] = led
            
            b_tit = ctk.CTkButton(top_cf, text=ch.name, width=60, height=22, corner_radius=11, font=APP_FONT_BOLD, fg_color=ROW_COLORS[i], hover_color="#888888", text_color="#000000")
            b_tit.pack(side="left", padx=2)
            self.widget_to_lfo_target[str(b_tit)] = (f"ch:{i}:trigger", f"{ch.name} Trigger")
            b_tit.bind("<ButtonPress-1>", lambda e, c=ch, b=b_tit: self._on_tit_press(c, b))
            b_tit.bind("<ButtonRelease-1>", lambda e, c=ch, b=b_tit: self._on_tit_release(c, b))
            b_tit.bind("<Button-3>", lambda e, c=ch: self.show_channel_swap_menu(e, c))
            
            btn_lock = ctk.CTkButton(top_cf, text="🔓", width=22, height=22, font=("Arial", 12), fg_color="#333333", hover_color="#555555")
            btn_lock.configure(command=lambda c=ch, b=btn_lock: self.toggle_lock(c, b)); btn_lock.pack(side="left", padx=2); self.ch_ui_refs[ch.name]['btn_lock'] = btn_lock
            
            btn_frame = ctk.CTkFrame(col, fg_color="transparent"); btn_frame.pack(fill="x", pady=1)
            btn_mute = ctk.CTkButton(btn_frame, text="M", width=25, font=("Trebuchet MS", 10, "bold"), fg_color="#444444"); btn_mute.configure(command=lambda c=ch, b=btn_mute: self.toggle_mute(c, b)); btn_mute.pack(side="left", padx=1); self.ch_ui_refs[ch.name]['btn_mute'] = btn_mute
            self.widget_to_lfo_target[str(btn_mute)] = (f"ch:{i}:mute", f"{ch.name} Mute")
            ToolTip(btn_mute, "Mute Channel\nLeft-click to toggle audio off/on for this track.", self)
            
            btn_solo = ctk.CTkButton(btn_frame, text="S", width=25, font=("Trebuchet MS", 10, "bold"), fg_color="#444444"); 
            btn_solo.configure(command=lambda c=ch, b=btn_solo: self.toggle_solo(c, b)); btn_solo.pack(side="left", padx=1); self.ch_ui_refs[ch.name]['btn_solo'] = btn_solo
            self.widget_to_lfo_target[str(btn_solo)] = (f"ch:{i}:solo", f"{ch.name} Solo")
            ToolTip(btn_solo, "Solo Channel\nLeft-click to isolate this track's audio.", self)
            
            btn_rev = ctk.CTkButton(btn_frame, text="R", width=25, font=("Trebuchet MS", 10, "bold"), fg_color="#444444")
            btn_rev.configure(command=lambda c=ch, b=btn_rev: self.toggle_rev(c, b)); btn_rev.pack(side="left", padx=1); self.ch_ui_refs[ch.name]['btn_rev'] = btn_rev
            self.widget_to_lfo_target[str(btn_rev)] = (f"ch:{i}:reverse", f"{ch.name} Reverse")
            ToolTip(btn_rev, "Reverse Sample\nLeft-click to play this track's sample backwards.", self)
            
            btn_cycl = ctk.CTkButton(btn_frame, text="Smpl", width=30, font=("Trebuchet MS", 10, "bold"), fg_color="#2ECC71", hover_color="#27AE60", text_color="#000000")
            btn_cycl.bind("<ButtonPress-1>", lambda e, c=ch, b=btn_cycl: self._on_cyc_press(c, b)); btn_cycl.bind("<ButtonRelease-1>", lambda e, c=ch, b=btn_cycl: self._on_cyc_release(c, b)); btn_cycl.bind("<Button-3>", lambda e, c=ch: self.cycle_drumb(c, False)); btn_cycl.pack(side="left", padx=1)
            self.widget_to_lfo_target[str(btn_cycl)] = (f"ch:{i}:cyc", f"{ch.name} Sample")
            
            self.ch_ui_refs[ch.name]['current_sample_idx'] = self.add_ch_sl(col, "Smpl", 0, 14, ch.current_sample_idx, lambda v, c=ch: self.handle_ch_slider(c, 'current_sample_idx', v), f"ch:{i}:current_sample_idx", f"{ch.name} Smpl Idx", is_int=True)
            self.ch_ui_refs[ch.name]['vol'] = self.add_ch_sl(col, "Vol", 0, 1, ch.vol, lambda v, c=ch: self.handle_ch_slider(c, 'vol', v), f"ch:{i}:vol", f"{ch.name} Vol")
            self.ch_ui_refs[ch.name]['pan'] = self.add_ch_sl(col, "Pan", -1, 1, ch.pan, lambda v, c=ch: self.handle_ch_slider(c, 'pan', v), f"ch:{i}:pan", f"{ch.name} Pan")
            self.ch_ui_refs[ch.name]['pitch'] = self.add_ch_sl(col, "Pitch", -24, 24, ch.pitch, lambda v, c=ch: self.handle_ch_slider(c, 'pitch', v), f"ch:{i}:pitch", f"{ch.name} Pitch", is_int=True)
            
            self.ch_ui_refs[ch.name]['sample'] = self.add_ch_range_sl(col, "Smpl Length", 0, 400, ch.sample_start, ch.sample_end, lambda v1, v2, c=ch: self.handle_ch_slider(c, 'sample', v1, True, v2), f"ch:{i}:sample_start", f"ch:{i}:sample_end", f"{ch.name} Smpl Len")
            self.ch_ui_refs[ch.name]['fade'] = self.add_ch_range_sl(col, "Fade", 0, 100, ch.fade_in, ch.fade_out, lambda v1, v2, c=ch: self.handle_ch_slider(c, 'fade', v1, True, v2), f"ch:{i}:fade_in", f"ch:{i}:fade_out", f"{ch.name} Fade")
            self.ch_ui_refs[ch.name]['euclid_k'] = self.add_ch_sl(col, "Euc", 0, self.engine.steps, ch.euclid_k, lambda v, c=ch: self.handle_ch_slider(c, 'euclid_k', v), f"ch:{i}:euclid_k", f"{ch.name} Euc", is_int=True)

            fx_container = ctk.CTkFrame(col, fg_color="transparent")
            fx_container.pack(fill="x", pady=2)
            self.ch_ui_refs[ch.name]['fx_container'] = fx_container
            self.ch_ui_refs[ch.name]['fx_knobs'] =[]

        self.seq_frame = ctk.CTkScrollableFrame(mixer_container, fg_color="#141414", orientation="vertical"); self.seq_frame.pack(fill="both", expand=True)

        for i in range(len(self.engine.fx_buses)):
            self.build_fx_rack_ui(i)
            for ch_idx, ch in enumerate(self.engine.channels):
                self.build_ch_fx_knob(ch, ch_idx, i)

        for i in range(len(self.engine.lfos)):
            self.build_lfo_ui(i)
            
        ctk.CTkButton(self.fx_frame, text="+ Add LFO", command=self.add_lfo, font=APP_FONT_BOLD).pack(pady=(0, 10))

        self.build_grid_ui()

    def set_juno_mode(self, bus_idx, mode):
        self.save_state()
        bus = self.engine.fx_buses[bus_idx]
        bus.juno_mode = mode
        if mode == "I":
            self.set_fx_param_from_ui(bus_idx, 0, 0.5, force_manual=False)
            self.set_fx_param_from_ui(bus_idx, 1, 1.5, force_manual=False)
            self.set_fx_param_from_ui(bus_idx, 2, 0.5, force_manual=False)
        elif mode == "II":
            self.set_fx_param_from_ui(bus_idx, 0, 0.83, force_manual=False)
            self.set_fx_param_from_ui(bus_idx, 1, 2.5, force_manual=False)
            self.set_fx_param_from_ui(bus_idx, 2, 0.5, force_manual=False)
        elif mode == "I+II":
            self.set_fx_param_from_ui(bus_idx, 0, 8.0, force_manual=False)
            self.set_fx_param_from_ui(bus_idx, 1, 1.0, force_manual=False)
            self.set_fx_param_from_ui(bus_idx, 2, 0.25, force_manual=False)

    def set_fx_param_from_ui(self, bus_idx, p_idx, val, force_manual=True):
        bus = self.engine.fx_buses[bus_idx]
        bus.set_param(p_idx + 1, val, force_manual)
        if p_idx < len(self.fx_ui_refs[bus_idx]['sliders']):
            sl_dict = self.fx_ui_refs[bus_idx]['sliders'][p_idx]
            sl_dict['knob'].set(val)
            self.update_fx_lbl_text(bus_idx, p_idx, val)

    def toggle_juno_bpm(self, bus_idx):
        self.save_state()
        bus = self.engine.fx_buses[bus_idx]
        bus.juno_bpm_sync = not bus.juno_bpm_sync

    def build_fx_rack_ui(self, i):
        color = FX_COLORS[i % len(FX_COLORS)]
        f_box = ctk.CTkFrame(self.fx_racks_inner, fg_color="#1A1A1A", border_width=1, border_color=color)
        f_box.pack(fill="x", padx=5, pady=3)
        
        # Left side for Global Send
        send_f = ctk.CTkFrame(f_box, fg_color="transparent")
        send_f.pack(side="left", padx=5, pady=2)
        
        lbl_send = ctk.CTkLabel(send_f, text=f"GLOBAL\n{self.engine.master_fx_sends[i]:.2f}", font=("Trebuchet MS", 9))
        def on_send_change(v, fi=i, l=lbl_send):
            self.engine.master_fx_sends[fi] = v
            l.configure(text=f"GLOBAL\n{v:.2f}")

        knob_send = CTkKnob(send_f, width=36, height=36, from_=0, to=1, command=on_send_change, progress_color=color)
        knob_send.set(self.engine.master_fx_sends[i])
        knob_send.pack()
        lbl_send.pack()

        t_id_send = f"gl:master_fx:{i}"
        self.bind_knob_events(knob_send, False, on_send_change, lbl_send, t_id_send, f"Master FX{i+1}", global_def=0.0)
        self.g_ui_refs[f"FX{i+1}"] = knob_send
        self.g_lbl_refs[f"FX{i+1}"] = lbl_send
        
        # Right side for Rack UI
        fx_f = ctk.CTkFrame(f_box, fg_color="transparent")
        fx_f.pack(side="left", fill="x", expand=True)

        header_f = ctk.CTkFrame(fx_f, fg_color="transparent")
        header_f.pack(fill="x", padx=5, pady=(2, 0))
        ctk.CTkLabel(header_f, text=f"FX{i+1}", font=APP_FONT_BOLD, text_color=color, height=20).pack(side="left")
        opt = ctk.CTkOptionMenu(header_f, values=list(FX_DEFS.keys()), font=APP_FONT, width=100, height=24, command=lambda v, idx=i: self.on_fx_type_change(idx, v))
        opt.set(self.engine.fx_buses[i].fx_type); opt.pack(side="right")

        knobs_f = ctk.CTkFrame(fx_f, fg_color="transparent")
        knobs_f.pack(fill="x", padx=2, pady=(0, 2))
        
        for c in range(12):
            knobs_f.grid_columnconfigure(c, weight=1)

        sl_refs =[]
        for j in range(5):
            k_frame = ctk.CTkFrame(knobs_f, fg_color="transparent")
            def on_change(v, b=i, p=j): 
                self.engine.fx_buses[b].set_param(p+1, v, force_manual=True)
                self.update_fx_lbl_text(b, p, v)
            knob = CTkKnob(k_frame, width=34, height=34, from_=0, to=1, command=on_change, progress_color=color)
            knob.pack(pady=(2,0))
            lbl = ctk.CTkLabel(k_frame, text="", font=("Trebuchet MS", 9))
            lbl.pack()
            self.bind_knob_events(knob, False, on_change, lbl, f"fx:{i}:p{j+1}", f"Master FX{i+1} P{j+1}", p_idx=j, bus_idx=i)
            sl_refs.append({'knob': knob, 'lbl': lbl, 'frame': k_frame})
        
        while len(self.fx_ui_refs) <= i: self.fx_ui_refs.append(None)
        self.fx_ui_refs[i] = {'opt': opt, 'sliders': sl_refs, 'box': f_box, 'knobs_f': knobs_f, 'fx_f': fx_f}
        self.update_fx_ui_labels(i)

    def build_ch_fx_knob(self, ch, ch_idx, fx_idx):
        color = FX_COLORS[fx_idx % len(FX_COLORS)]
        container = self.ch_ui_refs[ch.name]['fx_container']
        row_idx = fx_idx // 2
        rows = container.winfo_children()
        if row_idx < len(rows): row_f = rows[row_idx]
        else:
            row_f = ctk.CTkFrame(container, fg_color="transparent")
            row_f.pack(fill="x", pady=1)

        k_f = ctk.CTkFrame(row_f, fg_color="transparent")
        k_f.pack(side="left", expand=True)

        lbl = ctk.CTkLabel(k_f, text=f"FX{fx_idx+1}\n{ch.fx_sends[fx_idx]:.2f}", font=("Trebuchet MS", 9))
        def on_change(v, c=ch, fi=fx_idx): self.handle_ch_fx(c, fi, v)

        knob = CTkKnob(k_f, width=32, height=32, from_=0, to=1, command=on_change, progress_color=color)
        knob.set(ch.fx_sends[fx_idx])
        knob.pack()
        lbl.pack()

        self.bind_knob_events(knob, False, on_change, lbl, f"ch:{ch_idx}:fx_sends:{fx_idx}", f"{ch.name} FX{fx_idx+1}", global_def=0.0)
        self.ch_ui_refs[ch.name]['fx_knobs'].append((knob, lbl))

    def build_lfo_ui(self, i):
        f_box = ctk.CTkFrame(self.lfo_racks_inner, fg_color="#1A1A1A")
        f_box.pack(fill="x", padx=5, pady=2)
        
        lfo = self.engine.lfos[i]
        
        h_f = ctk.CTkFrame(f_box, fg_color="transparent"); h_f.pack(fill="x", padx=2, pady=1)
        ctk.CTkLabel(h_f, text=f"LFO {i+1}", font=("Trebuchet MS", 10, "bold"), text_color=lfo.color).pack(side="left")
        
        text = lfo.target_name if lfo.target_name else "[None]"
        font_size = 12
        if len(text) > 15: font_size = max(8, 12 - (len(text) - 15) // 2)
        lbl_targ = ctk.CTkLabel(h_f, text=text, font=("Trebuchet MS", font_size, "bold"), text_color="#FFFFFF")
        lbl_targ.pack(side="left", padx=5)
        
        btn_clear = ctk.CTkButton(h_f, text="❌", width=16, height=16, font=("Arial", 10), fg_color="#6B1D1D", command=lambda idx=i: self.clear_lfo_target(idx))
        if lfo.target_id: btn_clear.pack(side="right")
        
        r2 = ctk.CTkFrame(f_box, fg_color="transparent"); r2.pack(fill="x", padx=2, pady=1)
        
        opt_shp = ctk.CTkOptionMenu(r2, values=["Sine", "Triangle", "Square", "Random"], width=60, height=20, font=("Trebuchet MS", 9), command=lambda v, idx=i: setattr(self.engine.lfos[idx], 'shape', v))
        opt_shp.set(lfo.shape); opt_shp.pack(side="left", expand=True, padx=1)
        
        btn_sync = ctk.CTkButton(r2, text="BPM 🔒" if lfo.sync else "BPM 🔓", width=40, height=20, font=("Trebuchet MS", 9), fg_color="#2ECC71" if lfo.sync else "#333", text_color="#000000" if lfo.sync else "#FFFFFF", command=lambda idx=i: self.toggle_lfo_sync(idx))
        btn_sync.pack(side="left", expand=True, padx=1)
        
        opt_rate = ctk.CTkOptionMenu(r2, values=["8 Bar", "4 Bar", "2 Bar", "1 Bar", "1/2", "1/4", "1/8", "1/16"], width=50, height=20, font=("Trebuchet MS", 9), command=lambda v, idx=i: setattr(self.engine.lfos[idx], 'rate_sync', v))
        opt_rate.set(lfo.rate_sync)
        
        rate_k = CTkKnob(r2, width=28, height=28, from_=0.066, to=20.0, command=lambda v, idx=i: setattr(self.engine.lfos[idx], 'rate_hz', v))
        rate_k.set(lfo.rate_hz)
        
        depth_k = CTkKnob(r2, width=28, height=28, from_=0.0, to=1.0, command=lambda v, idx=i: setattr(self.engine.lfos[idx], 'depth', v))
        depth_k.set(lfo.depth)
        
        if lfo.sync: opt_rate.pack(side="left", expand=True, padx=1)
        else: rate_k.pack(side="left", expand=True, padx=1)
            
        depth_k.pack(side="left", expand=True, padx=1)
        ctk.CTkLabel(r2, text="Depth", font=("Trebuchet MS", 8)).pack(side="left", expand=True, padx=1)
        
        led = ctk.CTkFrame(r2, width=8, height=8, corner_radius=4, fg_color="#222"); led.pack(side="right", padx=4)
        
        while len(self.lfo_ui_refs) <= i: self.lfo_ui_refs.append(None)
        self.lfo_ui_refs[i] = {'lbl_target': lbl_targ, 'opt_shp': opt_shp, 'btn_sync': btn_sync, 'opt_rate': opt_rate, 'rate_k': rate_k, 'depth_k': depth_k, 'led': led, 'btn_clear': btn_clear}

    def add_fx_bus(self):
        self.save_state()
        new_idx = len(self.engine.fx_buses)
        self.engine.fx_buses.append(GlobalFXBus("Delay"))
        self.engine.master_fx_sends.append(0.0)
        self.engine.m_master_fx_sends.append(0.0)
        for ch in self.engine.channels:
            ch.fx_sends.append(0.0)
            ch.m_fx_sends.append(0.0)
            for p in ch.pattern_settings:
                if 'fx_sends' not in p:
                    p['fx_sends'] = [p.get('fx1',0), p.get('fx2',0), p.get('fx3',0)]
                p['fx_sends'].append(0.0)
                
        for v in self.engine.active_voices:
            v['fx_sends'].append(0.0)
        
        self.build_fx_rack_ui(new_idx)
        for i, ch in enumerate(self.engine.channels):
            self.build_ch_fx_knob(ch, i, new_idx)
            
    def add_lfo(self):
        idx = len(self.engine.lfos)
        colors = ["#00FFFF", "#FF00FF", "#FFFF00", "#00FF00", "#FFA500", "#FFC0CB", "#8A2BE2", "#00FA9A"]
        color = colors[idx % len(colors)]
        self.engine.lfos.append(LFO(color))
        self.build_lfo_ui(idx)

    def clear_lfo_target(self, idx):
        self.engine.lfos[idx].target_id = None; self.engine.lfos[idx].target_name = None
        self._update_lfo_target_label(idx, "[None]")
        self.lfo_ui_refs[idx]['btn_clear'].pack_forget()

    def toggle_lfo_sync(self, idx):
        lfo = self.engine.lfos[idx]
        lfo.sync = not lfo.sync
        btn = self.lfo_ui_refs[idx]['btn_sync']
        btn.configure(text="BPM 🔒" if lfo.sync else "BPM 🔓", fg_color="#2ECC71" if lfo.sync else "#333", text_color="#000000" if lfo.sync else "#FFFFFF")
        if lfo.sync:
            self.lfo_ui_refs[idx]['rate_k'].pack_forget()
            self.lfo_ui_refs[idx]['opt_rate'].pack(side="left", expand=True, padx=1, before=self.lfo_ui_refs[idx]['depth_k'])
        else:
            self.lfo_ui_refs[idx]['opt_rate'].pack_forget()
            self.lfo_ui_refs[idx]['rate_k'].pack(side="left", expand=True, padx=1, before=self.lfo_ui_refs[idx]['depth_k'])

    def _on_tit_press(self, ch, btn):
        setattr(btn, '_pressed', True); setattr(btn, '_held', False)
        t = threading.Timer(1.0, lambda c=ch, b=btn: self._on_tit_hold(c, b)); t.start(); setattr(btn, '_timer', t)

    def _on_tit_release(self, ch, btn):
        if getattr(btn, '_pressed', False):
            setattr(btn, '_pressed', False)
            if hasattr(btn, '_timer'): btn._timer.cancel()
            if not getattr(btn, '_held', False): self.engine.trigger_channel(ch)

    def _on_tit_hold(self, ch, btn):
        if getattr(btn, '_pressed', False):
            setattr(btn, '_held', True); btn.configure(text="Wait...")
            self.rescan_single_channel(ch, btn, restore_text=ch.name)

    def _on_cyc_press(self, ch, btn):
        setattr(btn, '_pressed', True); setattr(btn, '_held', False)
        t = threading.Timer(2.0, lambda c=ch, b=btn: self._on_cyc_hold(c, b)); t.start(); setattr(btn, '_timer', t)

    def _on_cyc_release(self, ch, btn):
        if getattr(btn, '_pressed', False):
            setattr(btn, '_pressed', False)
            if hasattr(btn, '_timer'): btn._timer.cancel()
            if not getattr(btn, '_held', False): self.cycle_drumb(ch)

    def _on_cyc_hold(self, ch, btn):
        if getattr(btn, '_pressed', False):
            setattr(btn, '_held', True); btn.configure(text="Wait...")
            self.rescan_single_channel(ch, btn, restore_text="Smpl")

    def handle_ch_slider(self, ch, param, val, is_range=False, val2=None):
        self.show_tip("Hold \"control\" while moving a slider to adjust all channels!")
        targets =[c for c in self.engine.channels if not c.locked] if self.ctrl_pressed else[ch]
        if ch not in targets and not ch.locked: targets.append(ch)
        
        for c in targets:
            if is_range:
                if param == 'sample':
                    c.sample_start, c.sample_end = val, val2
                    self.ch_ui_refs[c.name]['sample'][0].set(val, val2); self.ch_ui_refs[c.name]['sample'][1].configure(text=f"{int(val)}-{int(val2)}")
                elif param == 'fade':
                    c.fade_in, c.fade_out = val, val2
                    self.ch_ui_refs[c.name]['fade'][0].set(val, val2); self.ch_ui_refs[c.name]['fade'][1].configure(text=f"{int(val)}-{int(val2)}")
            elif param == 'euclid_k':
                self.apply_euclidean(c, int(val), refresh_ui=False, randomize=False)
                self.ch_ui_refs[c.name]['euclid_k'][0].set(int(val)); self.ch_ui_refs[c.name]['euclid_k'][1].configure(text=f"{int(val)}")
            else:
                setattr(c, param, val)
                ui_t = self.ch_ui_refs[c.name][param]
                ui_t[0].set(val)
                ui_t[1].configure(text=f"{val:.2f}" if param not in['pitch', 'current_sample_idx'] else f"{int(val)}")
        if param == 'euclid_k': self.refresh_grid_ui()

    def handle_ch_fx(self, ch, fx_idx, val):
        self.show_tip("Hold \"control\" while moving a slider to adjust all channels!")
        targets =[c for c in self.engine.channels if not c.locked] if self.ctrl_pressed else[ch]
        if ch not in targets and not ch.locked: targets.append(ch)
        for c in targets:
            c.fx_sends[fx_idx] = val
            k, l = self.ch_ui_refs[c.name]['fx_knobs'][fx_idx]
            k.set(val)
            l.configure(text=f"FX{fx_idx+1}\n{val:.2f}")

    def bind_knob_events(self, knob, is_int, command, lbl, t_id, t_name, p_idx=0, bus_idx=0, global_def=0.0):
        self.widget_to_lfo_target[str(knob)] = (t_id, t_name)
        self.widget_to_lfo_target[str(knob.canvas)] = (t_id, t_name)
        self.widget_to_lfo_target[str(lbl)] = (t_id, t_name)
        
        def target_updater(v=knob, tid=t_id):
            if v.is_dragging: return
            parts = tid.split(":")
            if parts[0] == "gl":
                if parts[1] == "master_fx": v.set(self.engine.m_master_fx_sends[int(parts[2])])
                else: v.set(getattr(self.engine, "m_"+parts[1]))
            elif parts[0] == "fx":
                v.set(getattr(self.engine.fx_buses[int(parts[1])], "m_"+parts[2]))
            elif parts[0] == "ch":
                if parts[2] == "fx_sends": v.set(self.engine.channels[int(parts[1])].m_fx_sends[int(parts[3])])
        self.target_updater_map[t_id] = target_updater
        
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
        lbl_f = ctk.CTkFrame(f, fg_color="transparent"); lbl_f.pack(fill="x", pady=0)
        lbl_name = ctk.CTkLabel(lbl_f, text=f"{label}:", font=("Trebuchet MS", 11), text_color="#AAAAAA")
        lbl_name.pack(side="left")
        lbl_val = ctk.CTkLabel(lbl_f, text=f"{int(def_start)}-{int(def_end)}", font=("Trebuchet MS", 11, "bold"), text_color="#3498DB")
        lbl_val.pack(side="right")
        
        def on_change(v1, v2): lbl_val.configure(text=f"{int(v1)}-{int(v2)}"); command(v1, v2)
        sl = CTkRangeSlider(f, width=130, height=20, from_=min_val, to=max_val, command=on_change); sl.set(def_start, def_end); sl.pack(anchor="w")
        for w in[sl, sl.canvas, lbl_name, lbl_val]:
            self.widget_to_lfo_target[str(w)] = (t_id_end, f"{t_name} End")
        
        ch_idx = int(t_id_start.split(":")[1])
        p_start = t_id_start.split(":")[2]
        p_end = t_id_end.split(":")[2]
        
        self.target_updater_map[t_id_start] = lambda s=sl, c=ch_idx, ps=p_start, pe=p_end: s.set(getattr(self.engine.channels[c], "m_"+ps), getattr(self.engine.channels[c], "m_"+pe)) if not s.is_dragging else None
        self.target_updater_map[t_id_end] = lambda s=sl, c=ch_idx, ps=p_start, pe=p_end: s.set(getattr(self.engine.channels[c], "m_"+ps), getattr(self.engine.channels[c], "m_"+pe)) if not s.is_dragging else None
        return sl, lbl_val

    def add_ch_sl(self, parent, label, min_val, max_val, default, command, t_id, t_name, is_int=False):
        f = ctk.CTkFrame(parent, fg_color="transparent"); f.pack(fill="x", padx=2, pady=0)
        lbl_f = ctk.CTkFrame(f, fg_color="transparent"); lbl_f.pack(fill="x", pady=0)
        lbl_name = ctk.CTkLabel(lbl_f, text=f"{label}:", font=("Trebuchet MS", 11), text_color="#AAAAAA")
        lbl_name.pack(side="left")
        
        val_str = f"{default:.2f}" if not is_int else f"{int(default)}"
        lbl_val = ctk.CTkLabel(lbl_f, text=val_str, font=("Trebuchet MS", 11, "bold"), text_color="#3498DB")
        lbl_val.pack(side="right")
        
        def on_change(v):
            val = int(v) if is_int else v; lbl_val.configure(text=f"{val:.2f}" if not is_int else f"{val}"); command(val)
        sl = ctk.CTkSlider(f, from_=min_val, to=max_val, width=130, command=on_change); sl.set(default); sl.pack(anchor="w")
        
        ch_idx, param = int(t_id.split(":")[1]), t_id.split(":")[2]
        self.target_updater_map[t_id] = lambda s=sl, c=ch_idx, p=param: s.set(getattr(self.engine.channels[c], "m_"+p))
        
        for w in [sl, sl._canvas, lbl_name, lbl_val]: self.widget_to_lfo_target[str(w)] = (t_id, t_name)
        
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
        lbl_name.bind("<Double-Button-1>", manual_entry)
        lbl_val.bind("<Double-Button-1>", manual_entry)
        for w in[sl, sl._canvas]: w.bind("<Double-Button-1>", reset); w.bind("<MouseWheel>", wheel); w.bind("<Button-4>", wheel); w.bind("<Button-5>", wheel)
        return sl, lbl_val

    def on_fx_type_change(self, bus_idx, new_type):
        self.save_state(); self.engine.fx_buses[bus_idx].set_type(new_type); self.update_fx_ui_labels(bus_idx)

    def update_fx_ui_labels(self, bus_idx):
        bus = self.engine.fx_buses[bus_idx]; defs = FX_DEFS[bus.fx_type]; ui = self.fx_ui_refs[bus_idx]
        ui['opt'].set(bus.fx_type)
        
        if ui.get('custom_ui') and ui['custom_ui'].winfo_exists():
            ui['custom_ui'].destroy()
            ui['custom_ui'] = None

        if bus.fx_type == "Juno Chorus":
            custom_f = ctk.CTkFrame(ui['fx_f'], fg_color="transparent")
            custom_f.pack(fill="x", padx=5, pady=(0, 2), before=ui['knobs_f'])
            ui['custom_ui'] = custom_f
            
            btn_f = ctk.CTkFrame(custom_f, fg_color="transparent")
            btn_f.pack(side="left")
            
            ui['juno_btns'] = {}
            for m in ["I", "II", "I+II", "MANUAL"]:
                b = ctk.CTkButton(btn_f, text=m, width=28, height=20, font=("Trebuchet MS", 9, "bold"), 
                                  fg_color="#D35400" if bus.juno_mode == m else "#333",
                                  command=lambda mode=m, idx=bus_idx: self.set_juno_mode(idx, mode))
                b.pack(side="left", padx=1)
                ui['juno_btns'][m] = b
                
            bpm_btn = ctk.CTkButton(custom_f, text="🎵", width=24, height=20, font=("Arial", 12),
                                    fg_color="#2ECC71" if bus.juno_bpm_sync else "#333", text_color="#000",
                                    command=lambda b=bus_idx: self.toggle_juno_bpm(b))
            bpm_btn.pack(side="right", padx=5)
            ui['juno_bpm_btn'] = bpm_btn
            ToolTip(bpm_btn, "BPM Sync Rate", self)
        
        # Identify active knobs
        active_j = [j for j in range(5) if j < len(defs) and defs[j][0]]
        num_knobs = len(active_j)
        
        if num_knobs == 4:
            row_sizes = [4]
        elif num_knobs > 6:
            row_sizes = [4] * (num_knobs // 4) + ([num_knobs % 4] if num_knobs % 4 else[])
        else:
            row_sizes =[]
            rem = num_knobs
            while rem > 0:
                take = min(3, rem)
                row_sizes.append(take)
                rem -= take
        
        # Hide all slider frames first
        for sl in ui['sliders']:
            sl['frame'].grid_forget()
            
        knob_idx = 0
        for row_idx, r_size in enumerate(row_sizes):
            for k in range(r_size):
                sl_dict = ui['sliders'][active_j[knob_idx]]
                p_idx = active_j[knob_idx]
                p_def = defs[p_idx]
                
                # Compute grid column mathematically to flawlessly center the row components
                if r_size == 1:
                    col = 0; colspan = 12
                elif r_size == 2:
                    col = 2 if k == 0 else 6; colspan = 4
                elif r_size == 3:
                    col = k * 4; colspan = 4
                else: # r_size == 4
                    col = k * 3; colspan = 3
                    
                sl_dict['frame'].grid(row=row_idx, column=col, columnspan=colspan, pady=1, sticky="n")
                sl_dict['knob'].is_fx_rack = True
                sl_dict['knob'].configure_range(p_def[1], p_def[2])
                val = getattr(bus, f"p{p_idx+1}")
                sl_dict['knob'].set(val)
                self.update_fx_lbl_text(bus_idx, p_idx, val)
                
                knob_idx += 1

    def update_fx_lbl_text(self, bus_idx, p_idx, val):
        bus = self.engine.fx_buses[bus_idx]; defs = FX_DEFS[bus.fx_type]
        if p_idx < len(defs) and defs[p_idx][0]:
            p_name = defs[p_idx][0]
            if bus.fx_type == "Resonant LPF" and p_idx == 0: self.fx_ui_refs[bus_idx]['sliders'][p_idx]['lbl'].configure(text=f"{p_name}\n{int(val)} Hz")
            else: self.fx_ui_refs[bus_idx]['sliders'][p_idx]['lbl'].configure(text=f"{p_name}\n{val:.2f}")

    def change_audio_device(self, dev_label):
        if hasattr(self, 'audio_dev_map') and dev_label in self.audio_dev_map:
            self.engine.set_audio_device(self.audio_dev_map[dev_label])

    def change_midi_device(self, dev_name):
        self.engine.set_midi_device(dev_name)

    def toggle_midi_sync(self):
        self.engine.midi_sync = self.midi_sync_var.get()
        if self.engine.midi_sync and self.engine.midi_port is None and getattr(self.engine, 'midi_port_name', None) and self.engine.midi_port_name != "None":
            self.engine.set_midi_device(self.engine.midi_port_name)

    def open_options(self):
        if hasattr(self, 'options_window') and self.options_window is not None and self.options_window.winfo_exists():
            self.options_window.focus()
            return
            
        w = ctk.CTkToplevel(self); w.title("Options"); w.geometry("400x480"); w.transient(self)
        self.options_window = w
        ctk.CTkLabel(w, text="Settings", font=APP_FONT_BOLD).pack(pady=10)
        
        ctk.CTkLabel(w, text="Scan Method:", font=APP_FONT).pack(pady=(10, 0))
        ctk.CTkSegmentedButton(w, values=["Lazy", "Normal", "Focused"], variable=self.scan_method_var).pack(pady=5)
        
        ctk.CTkCheckBox(w, text="Auto-detect BPM on load", font=APP_FONT, variable=self.auto_detect_bpm).pack(pady=5)
        ctk.CTkCheckBox(w, text="Show Contextual Tips", font=APP_FONT, variable=self.show_tips_var).pack(pady=5)
        ctk.CTkCheckBox(w, text="Enable Hover Tooltips", font=APP_FONT, variable=self.enable_hover_tooltips).pack(pady=5)
        
        self.audio_dev_map = {}
        hostapis = sd.query_hostapis()
        for i, d in enumerate(sd.query_devices()):
            if d['max_output_channels'] > 0:
                api_name = hostapis[d['hostapi']]['name']
                self.audio_dev_map[f"{d['name']}[{api_name}]"] = i
                
        out_devs = list(self.audio_dev_map.keys())
        if out_devs:
            ctk.CTkLabel(w, text="Audio Output Device", font=APP_FONT).pack(pady=(10, 0))
            curr_idx = getattr(self.engine, 'current_device_idx', sd.default.device[1])
            curr_val = out_devs[0]
            for lbl, idx in self.audio_dev_map.items():
                if idx == curr_idx:
                    curr_val = lbl
                    break
            self.audio_dev_var = ctk.StringVar(value=curr_val)
            ctk.CTkOptionMenu(w, values=out_devs, variable=self.audio_dev_var, font=APP_FONT, command=self.change_audio_device).pack(pady=5)
            
        if HAS_MIDO:
            ctk.CTkLabel(w, text="MIDI Sync Settings", font=APP_FONT_BOLD).pack(pady=(10, 0))
            midi_in_devs = mido.get_input_names()
            if midi_in_devs:
                midi_in_devs = list(dict.fromkeys(midi_in_devs))
                self.midi_dev_var = ctk.StringVar(value=getattr(self.engine, 'midi_port_name', None) if getattr(self.engine, 'midi_port_name', None) else "None")
                opts = ["None"] + midi_in_devs
                ctk.CTkOptionMenu(w, values=opts, variable=self.midi_dev_var, font=APP_FONT, command=self.change_midi_device).pack(pady=5)
                
                self.midi_sync_var = ctk.BooleanVar(value=getattr(self.engine, 'midi_sync', False))
                ctk.CTkCheckBox(w, text="Enable External MIDI Sync", font=APP_FONT, variable=self.midi_sync_var, command=self.toggle_midi_sync).pack(pady=5)
            else:
                ctk.CTkLabel(w, text="No MIDI input devices found.", font=APP_FONT).pack(pady=5)
        else:
            ctk.CTkLabel(w, text="MIDI sync unavailable (mido not installed)", font=APP_FONT).pack(pady=10)

    def sync_ui_to_engine(self):
        while len(self.fx_ui_refs) < len(self.engine.fx_buses):
            new_idx = len(self.fx_ui_refs)
            self.build_fx_rack_ui(new_idx)
            for i, ch in enumerate(self.engine.channels):
                self.build_ch_fx_knob(ch, i, new_idx)

        for p, k, is_int in[('Vol','global_vol',False),('Pitch','global_pitch',True),('Swing','swing',True)]:
            val = getattr(self.engine, k); self.g_ui_refs[p].set(val)
            self.g_lbl_refs[p].configure(text=f"{p}\n{val:.2f}" if not is_int else f"{p}\n{int(val)}")
            
        for i, send in enumerate(self.engine.master_fx_sends):
            k = f"FX{i+1}"
            self.g_ui_refs[k].set(send)
            self.g_lbl_refs[k].configure(text=f"GLOBAL\n{send:.2f}")

        self.bpm_widget.set_val(self.engine.bpm); self.pat_var.set(str(self.engine.current_pattern)); self.step_var.set(str(self.engine.steps))
        
        if "gl:stutter_div_idx" in self.target_updater_map:
            self.target_updater_map["gl:stutter_div_idx"]()
        
        for ch in self.engine.channels:
            r = self.ch_ui_refs[ch.name]
            r['btn_mute'].configure(fg_color="#CC0000" if ch.mute else "#444444")
            r['btn_lock'].configure(text="🔒", text_color="#E74C3C") if ch.locked else r['btn_lock'].configure(text="🔓", text_color="#FFFFFF")
            r['btn_solo'].configure(fg_color="#CCCC00" if ch.solo else "#444444")
            r['btn_rev'].configure(fg_color="#9B59B6" if ch.reverse else "#444444")
            
            for p, attr, is_int in[('current_sample_idx','current_sample_idx',True), ('vol','vol',False),('pan','pan',False),('pitch','pitch',True),('euclid_k','euclid_k',True)]:
                sl, lbl_val = r[p]; v = getattr(ch, attr); sl.set(v)
                lbl_val.configure(text=f"{v:.2f}" if not is_int else f"{int(v)}")
            r['euclid_k'][0].configure(to=self.engine.steps)
            r['sample'][1].configure(text=f"{int(ch.sample_start)}-{int(ch.sample_end)}")
            r['fade'][1].configure(text=f"{int(ch.fade_in)}-{int(ch.fade_out)}")
            
            for fx_idx, val in enumerate(ch.fx_sends):
                k, l = r['fx_knobs'][fx_idx]
                k.set(val)
                l.configure(text=f"FX{fx_idx+1}\n{val:.2f}")

        for i in range(len(self.engine.fx_buses)): self.update_fx_ui_labels(i)
        for i in range(len(self.engine.lfos)):
            lfo = self.engine.lfos[i]
            if lfo.target_id: self.lfo_ui_refs[i]['btn_clear'].pack(side="right")
            else: self.lfo_ui_refs[i]['btn_clear'].pack_forget()

        self.build_grid_ui() if len(self.grid_buttons) != 8 * self.engine.steps else self.refresh_grid_ui()

    def reset_all_params(self, e=None):
        self.save_state()
        self.engine.apply_state({
            'bpm': 75, 'steps': 16, 'pat': 0, 'swing': 10, 'gvol': 0.8, 'gpitch': 0, 'm_fx_sends': [0.0]*len(self.engine.fx_buses), 'stutter_div_idx': 4,
            'ch':[{'idx': 0, 'mute': False, 'solo_l': False, 'locked': False, 'rev': False, 'vol': 0.8, 'pan': 0.0, 'pitch': 0, 's_start': 0, 's_end': 400, 'fx_sends': [0.0]*len(self.engine.fx_buses), 'euc': 0, 'seq': [[False]*32 for _ in range(4)], 'ps':[{'vol':0.8, 'pan':0.0, 'pitch':0, 's_start':0, 's_end':400, 'fx_sends':[0.0]*len(self.engine.fx_buses), 'euclid_k':0, 'fade_in':1.0, 'fade_out':100.0, 'smpl_idx':0} for _ in range(4)]} for _ in range(8)],
            'fx':[{'type': 'Reverb', 'p1': 0.2, 'p2': 0.2, 'p3': 0.0, 'p4': 0.0, 'p5': 0.0}, {'type': 'Resonant LPF', 'p1': 20000.0, 'p2': 0.0, 'p3': 0.0, 'p4': 0.0, 'p5': 0.0}, {'type': 'Compressor', 'p1': 0.1, 'p2': 4.0, 'p3': 0.05, 'p4': 0.01, 'p5': 0.1}] + [{'type': 'Delay', 'p1': 0.3, 'p2': 0.4, 'p3': 0.0, 'p4': 0.0, 'p5': 0.0} for _ in range(len(self.engine.fx_buses)-3)]
        }); self.sync_ui_to_engine(); self.lbl_status.configure(text="All parameters reset.")

    def load_file(self):
        if fp := filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")]):
            self.current_audio_file = fp; self.lbl_status.configure(text="Extracting drumbs... Please wait.")
            self.is_scanning = True
            self.animate_scan()
            threading.Thread(target=self.process_audio_file, args=(fp, False), daemon=True).start()

    def show_no_sample_overlay(self):
        if hasattr(self, 'warning_frame') and self.warning_frame.winfo_exists(): return
        self.warning_frame = ctk.CTkFrame(self, fg_color="#111111", corner_radius=10, border_width=2, border_color="#E74C3C")
        self.warning_lbl = ctk.CTkLabel(self.warning_frame, text="Load & scan a sample to get started!", font=("Trebuchet MS", 34, "bold"), text_color="#E74C3C")
        self.warning_lbl.pack(padx=20, pady=15)
        self.warning_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.fade_warning_overlay(0)

    def fade_warning_overlay(self, step):
        if not hasattr(self, 'warning_frame') or not self.warning_frame.winfo_exists(): return
        max_steps = 40
        if step >= max_steps:
            self.warning_frame.place_forget()
            self.warning_frame.destroy()
            return
            
        r = int(231 - (231 - 10) * (step / max_steps))
        g = int(76 - (76 - 10) * (step / max_steps))
        b = int(60 - (60 - 10) * (step / max_steps))
        color = f"#{r:02x}{g:02x}{b:02x}"
        
        self.warning_lbl.configure(text_color=color)
        self.warning_frame.configure(border_color=color)
        self.after(100, lambda: self.fade_warning_overlay(step + 1))

    def rescan_file(self):
        if self.current_audio_file:
            self.lbl_status.configure(text="Rescanning for fresh drumbs...")
            self.is_scanning = True
            self.animate_scan()
            threading.Thread(target=self.process_audio_file, args=(self.current_audio_file, True), daemon=True).start()
        else: self.lbl_status.configure(text="Load a file first before rescanning!")

    def rescan_single_channel(self, ch, btn, restore_text="Smpl"):
        if not self.current_audio_file:
            self.after(0, lambda:[btn.configure(text=restore_text), self.lbl_status.configure(text="No file loaded to rescan!")]); return
        def task():
            try:
                ch.samples = self.extractor.rescan_single_channel(self.current_audio_file, ch.name)
                ch.current_sample_idx = 0
                self.after(0, lambda: self.lbl_status.configure(text=f"Rescanned new samples for {ch.name}"))
            except Exception as e: self.after(0, lambda e=e: self.lbl_status.configure(text=f"Error: {e}"))
            finally: self.after(0, lambda: btn.configure(text=restore_text))
        threading.Thread(target=task, daemon=True).start()

    def process_audio_file(self, filepath, is_rescan=False):
        is_initial = not self.first_load_done
        try:
            drumbs, tempo = self.extractor.extract(filepath, method=self.scan_method_var.get(), is_rescan=is_rescan)
            for ch in self.engine.channels:
                if ch.name in drumbs: ch.samples = drumbs[ch.name]
            
            def _update_gui():
                self.is_scanning = False
                if is_initial:
                    self.first_load_done = True; self.save_state()
                    self.refresh_grid_ui()
                if self.auto_detect_bpm.get() and tempo > 0:
                    self.engine.bpm = max(40, min(300, float(tempo)))
                    self.bpm_widget.set_val(self.engine.bpm)
                self.btn_rescan.configure(state="normal", fg_color="#B9770E")
                self.btn_play.configure(state="normal", fg_color="#228B22")
                self.lbl_status.configure(text=f"Extraction complete!")
                
            self.after(0, _update_gui)
        except Exception as e: self.after(0, lambda e=e: [setattr(self, 'is_scanning', False), self.lbl_status.configure(text=f"Error: {str(e)}")])

    def toggle_play(self):
        if not self.first_load_done:
            self.show_no_sample_overlay()
            return
            
        self.engine.is_playing = not self.engine.is_playing
        if self.engine.is_playing: 
            self.engine.step_counter = 0; self.engine.samples_until_next_step = 0.0
            self.engine.current_step = 0
            self.btn_play.configure(text="⏸", fg_color="#228B22")
            if self.btn_rec.cget("state") == "disabled":
                self.btn_rec.configure(state="normal", fg_color="#CC0000")
        else:
            self.btn_play.configure(text="▶", fg_color="#228B22")
            if self.engine.is_recording:
                self.toggle_record()
            self.btn_rec.configure(state="disabled", fg_color="#333333")

    def toggle_record(self):
        if not self.engine.is_recording:
            self.engine.record_buffer =[]; self.engine.is_recording = True; self.btn_rec.configure(fg_color="#8B0000", text_color="#FFFFFF")
        else:
            self.engine.is_recording = False; self.btn_rec.configure(fg_color="#CC0000", text_color="#FFFFFF")
            if self.engine.record_buffer:
                audio_data = np.concatenate(self.engine.record_buffer, axis=0)
                if fp := filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")]): wavfile.write(fp, SAMPLE_RATE, audio_data); self.lbl_status.configure(text=f"Recording saved.")

    def unlock_all(self, e=None):
        if any(ch.locked for ch in self.engine.channels):
            self.lock_history.append({ch.name: ch.locked for ch in self.engine.channels})
            for ch in self.engine.channels:
                ch.locked = False
                self.ch_ui_refs[ch.name]['btn_lock'].configure(text="🔓", text_color="#FFFFFF")

    def undo_unlock_all(self, e=None):
        if not self.lock_history: return
        state = self.lock_history.pop()
        for ch in self.engine.channels:
            if ch.name in state:
                ch.locked = state[ch.name]
                btn = self.ch_ui_refs[ch.name]['btn_lock']
                btn.configure(text="🔒", text_color="#E74C3C") if ch.locked else btn.configure(text="🔓", text_color="#FFFFFF")

    def unsolo_all(self, e=None):
        if any(ch.solo or ch.solo_locked for ch in self.engine.channels):
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
                if ch.solo: self.ch_ui_refs[ch.name]['btn_solo'].configure(fg_color="#CCCC00")
                else: self.ch_ui_refs[ch.name]['btn_solo'].configure(fg_color="#444444")
            
    def unmute_all(self, e=None):
        if any(ch.mute for ch in self.engine.channels):
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

    def unreverse_all(self, e=None):
        if any(ch.reverse for ch in self.engine.channels):
            self.rev_history.append({ch.name: ch.reverse for ch in self.engine.channels})
            for ch in self.engine.channels:
                ch.reverse = False
                self.ch_ui_refs[ch.name]['btn_rev'].configure(fg_color="#444444")

    def undo_unreverse_all(self, e=None):
        if not self.rev_history: return
        state = self.rev_history.pop()
        for ch in self.engine.channels:
            if ch.name in state:
                ch.reverse = state[ch.name]
                self.ch_ui_refs[ch.name]['btn_rev'].configure(fg_color="#9B59B6" if ch.reverse else "#444444")

    def toggle_mute(self, ch, btn): ch.mute = not ch.mute; btn.configure(fg_color="#CC0000" if ch.mute else "#444444")
    
    def toggle_rev(self, ch, btn):
        ch.reverse = not ch.reverse; btn.configure(fg_color="#9B59B6" if ch.reverse else "#444444")
        
    def toggle_lock(self, ch, btn): 
        ch.locked = not ch.locked
        btn.configure(text="🔒", text_color="#E74C3C") if ch.locked else btn.configure(text="🔓", text_color="#FFFFFF")

    def toggle_solo(self, ch, btn):
        ch.solo_locked = not ch.solo_locked
        ch.solo = ch.solo_locked
        btn.configure(fg_color="#CCCC00" if ch.solo else "#444444")

    def cycle_drumb(self, ch, forward=True): 
        ch.step_sample(forward)
        if ch.name in self.ch_ui_refs:
            self.ch_ui_refs[ch.name]['current_sample_idx'][0].set(ch.current_sample_idx)
            self.ch_ui_refs[ch.name]['current_sample_idx'][1].configure(text=f"{ch.current_sample_idx}")
        self.lbl_status.configure(text=f"{ch.name} -> drumb {ch.current_sample_idx + 1}/15")

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
        self.apply_euclidean(ch, val, randomize=True); self.ch_ui_refs[ch.name]['euclid_k'][0].set(val); self.ch_ui_refs[ch.name]['euclid_k'][1].configure(text=f"{val}")

    def rand_euc(self):
        for ch in self.engine.channels:
            if not ch.locked:
                val = self.get_weighted_euc_val(self.engine.steps)
                self.apply_euclidean(ch, val, randomize=True); self.ch_ui_refs[ch.name]['euclid_k'][0].set(val); self.ch_ui_refs[ch.name]['euclid_k'][1].configure(text=f"{val}")

    def reset_rand_sam(self, e=None):
        self.save_state()
        for ch in self.engine.channels:
            if not ch.locked: self.handle_ch_slider(ch, 'sample', 0, True, 400)

    def reset_rand_fade(self, e=None):
        self.save_state()
        for ch in self.engine.channels:
            if not ch.locked:
                self.handle_ch_slider(ch, 'fade', 1.0, True, 100.0)

    def rand_sam(self):
        for ch in self.engine.channels:
            if not ch.locked:
                s_end = random.uniform(100, 400); s_start = random.uniform(0, max(0, s_end - 50))
                self.handle_ch_slider(ch, 'sample', s_start, True, s_end)

    def rand_fade(self):
        for ch in self.engine.channels:
            if not ch.locked:
                f_in = random.uniform(0, 40)
                f_out = random.uniform(60, 100)
                self.handle_ch_slider(ch, 'fade', f_in, True, f_out)

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
            
    def rand_all(self): self.rand_cycl(); self.rand_euc(); self.rand_sam(); self.rand_fade(); self.rand_pan()

    def change_pattern(self, val):
        self.save_state()
        self.show_tip("If you choose a new pattern and it is blank, it will copy your existing pattern!")
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
            self.ch_ui_refs[ch.name]['euclid_k'][0].configure(to=val)
            if ch.euclid_k > val:
                self.handle_ch_slider(ch, 'euclid_k', val)

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
        self.widget_to_step.clear(); self.grid_buttons.clear(); self.grid_bgs.clear()
        
        steps = self.engine.steps; pat = self.engine.current_pattern
        inner_container = ctk.CTkFrame(self.seq_frame, fg_color="transparent"); inner_container.pack(fill="x", anchor="n")
        
        for i, ch in enumerate(self.engine.channels):
            row_frame = ctk.CTkFrame(inner_container, fg_color="transparent"); row_frame.pack(fill="x", pady=1) 
            title_f = ctk.CTkFrame(row_frame, fg_color="transparent", width=90, height=32); title_f.pack(side="left", padx=5); title_f.pack_propagate(False)
            ctk.CTkLabel(title_f, text=ch.name, anchor="e", font=APP_FONT_BOLD).pack(side="left", padx=2, fill="x", expand=True)
            
            btn_dice = ctk.CTkButton(title_f, text="🎲", width=22, height=22, font=("Arial", 12), fg_color="#333333", hover_color="#555555", command=lambda c=ch: self.roll_single_euc(c))
            btn_dice.pack(side="right")
            self.widget_to_lfo_target[str(btn_dice)] = (f"ch:{i}:roll_euc", f"{ch.name} Roll Euc")
            
            for j in range(steps):
                bg_col = "#151515" if (j // 4) % 2 == 0 else "#252525" 
                step_w = 32 if steps <= 16 else 15
                step_bg = ctk.CTkFrame(row_frame, fg_color=bg_col, corner_radius=6, height=32, width=step_w, border_width=2, border_color=bg_col)
                step_bg.pack(side="left", padx=1, expand=True, fill="both"); step_bg.pack_propagate(False)
                state = ch.sequence[pat][j]
                btn = ctk.CTkButton(step_bg, text="", corner_radius=4, fg_color=ROW_COLORS[i] if state else "transparent", hover_color=ROW_COLORS[i], border_width=1, border_color="#333333", command=lambda c=i, s=j: self.toggle_step(c, s, not self.engine.channels[c].sequence[self.engine.current_pattern][s]))
                btn.place(relx=0.5, rely=0.5, relwidth=0.8, relheight=0.8, anchor="center")
                self.grid_buttons[(i, j)] = btn
                self.grid_bgs[(i, j)] = step_bg
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
        if 'engine' in locals():
            engine.stream.stop()
            engine.stream.close()
            if getattr(engine, 'midi_port', None):
                engine.midi_port.close()