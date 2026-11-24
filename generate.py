#!/usr/bin/env python3
"""
Meow Music Generator - 第一版测试脚本
输入：包含猫叫声的音频/视频文件
输出：output.mp3（猫叫旋律 + 弦乐伴奏）
"""

import sys
import os
import math
import random
import subprocess
import tempfile
import numpy as np
import librosa
import soundfile as sf
from midiutil import MIDIFile

# === 配置 ===
SOUNDFONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GeneralUser-GS.sf2")
SR = 22050
OUTPUT_DURATION = 24  # 秒

# 风格配置表：每种风格定义完整的音乐参数
GENRES = {
    'folk': {
        'label': '民谣',
        'bpm': 90,
        'beats_per_chord': 4,
        'chords': [[48,52,55],[55,59,62],[57,60,64],[53,57,60]],  # C G Am F
        'preferred_dur': (0.25, 0.80),
    },
    'lofi': {
        'label': 'Lo-fi',
        'bpm': 75,
        'beats_per_chord': 4,
        'chords': [[60,64,67,71],[57,60,64,67],[52,55,59,62],[53,57,60,64]],  # Cmaj7 Am7 Em7 Fmaj7
        'preferred_dur': (0.30, 0.90),
    },
    'rb': {
        'label': 'R&B',
        'bpm': 82,
        'beats_per_chord': 4,
        'chords': [[60,64,67,71],[57,60,64,67],[53,57,60,64],[55,59,62,65]],  # Cmaj7 Am7 Fmaj7 G7
        'preferred_dur': (0.40, 1.20),
    },
    'jazz': {
        'label': '爵士',
        'bpm': 108,
        'beats_per_chord': 4,
        'chords': [[50,53,57,60],[55,59,62,65],[48,52,55,59],[57,60,64,67]],  # Dm7 G7 Cmaj7 Am7
        'preferred_dur': (0.20, 0.60),
    },
    'classical': {
        'label': '古典小品',
        'bpm': 72,
        'beats_per_chord': 4,
        'chords': [[48,52,55],[57,60,64],[50,53,57],[55,59,62]],  # C Am Dm G
        'preferred_dur': (0.30, 0.80),
    },
    'march': {
        'label': '进行曲',
        'bpm': 112,
        'beats_per_chord': 2,
        'chords': [[48,52,55],[53,57,60],[55,59,62],[48,52,55]],  # C F G C
        'preferred_dur': (0.10, 0.40),
    },
}

# 运行时全局（由 apply_genre 更新）
BPM = 90
BEATS_PER_CHORD = 4
CHORD_PROGRESSION = GENRES['folk']['chords']
GENRE_KEY = 'folk'


def apply_genre(key):
    global BPM, BEATS_PER_CHORD, CHORD_PROGRESSION, GENRE_KEY
    g = GENRES[key]
    BPM, BEATS_PER_CHORD, CHORD_PROGRESSION, GENRE_KEY = g['bpm'], g['beats_per_chord'], g['chords'], key


def recommend_genres(segments, n=4):
    """根据猫叫平均时长给风格打分，返回最适合的前 n 个"""
    durations = [e - s for s, e, *_ in segments]
    avg = float(np.mean(durations))
    scores = []
    for key, g in GENRES.items():
        lo, hi = g['preferred_dur']
        mid = (lo + hi) / 2
        if lo <= avg <= hi:
            score = 1.0 - abs(avg - mid) / (hi - lo + 1e-9)
        else:
            score = max(0.0, 1.0 - min(abs(avg - lo), abs(avg - hi)) * 3)
        scores.append((key, score))
    scores.sort(key=lambda x: -x[1])
    return [(k, s) for k, s in scores[:n]]


# === 根据猫叫音高生成和弦走向 ===
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # natural minor

# 每个风格的和声走向用「级数」表达（0=I, 1=ii, ... 6=vii），移调到任意调都保持性格
GENRE_DEGREES = {
    'folk':      {'deg': [0, 4, 5, 3], 'seventh': False},  # I  V  vi IV
    'lofi':      {'deg': [0, 5, 2, 3], 'seventh': True},    # I  vi iii IV
    'rb':        {'deg': [0, 5, 3, 4], 'seventh': True},    # I  vi IV V
    'jazz':      {'deg': [1, 4, 0, 5], 'seventh': True},    # ii V  I  vi
    'classical': {'deg': [0, 5, 1, 4], 'seventh': False},   # I  vi ii V
    'march':     {'deg': [0, 3, 4, 0], 'seventh': False},   # I  IV V  I
}

PC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']


def detect_key(note_events):
    """从猫叫音高分布侦测调性，返回 (tonic 0-11, mode 'maj'/'min')"""
    hist = [0.0] * 12
    for ev in note_events:
        hist[ev['midi'] % 12] += 1

    best = None
    for tonic in range(12):
        for mode, scale, triad in [('maj', MAJOR_SCALE, [0, 4, 7]),
                                    ('min', MINOR_SCALE, [0, 3, 7])]:
            scale_pcs = {(tonic + o) % 12 for o in scale}
            triad_pcs = {(tonic + o) % 12 for o in triad}
            score = 0.0
            for pc in range(12):
                if pc in scale_pcs:
                    score += hist[pc]
                if pc in triad_pcs:
                    score += hist[pc] * 0.5   # 主和弦音加权，稳住调式中心
            if best is None or score > best[0]:
                best = (score, tonic, mode)
    return best[1], best[2]


def build_generated_progression(genre_key, note_events):
    """按侦测出的调 + 风格级数模板，生成一条和弦走向（返回 MIDI 音符列表）"""
    tonic, mode = detect_key(note_events)
    scale = MAJOR_SCALE if mode == 'maj' else MINOR_SCALE
    base = 48 + tonic  # 根音八度：C3=48 起

    # 两个八度的音阶 MIDI，方便按级数堆三度
    scale_midi = ([base + o for o in scale] +
                  [base + 12 + o for o in scale] +
                  [base + 24 + o for o in scale])

    tmpl = GENRE_DEGREES[genre_key]
    progression, romans = [], []
    for d in tmpl['deg']:
        idxs = [d, d + 2, d + 4] + ([d + 6] if tmpl['seventh'] else [])
        chord = [scale_midi[i] for i in idxs]
        progression.append(chord)
        rn = ROMAN[d % 7]
        romans.append(rn if mode == 'maj' else rn.lower())

    key_name = f"{PC_NAMES[tonic]} {'大调' if mode == 'maj' else '小调'}"
    print(f"🎼 侦测调性：{key_name}　生成走向：{' - '.join(romans)}")
    return progression


def hz_to_midi(hz):
    if hz is None or hz <= 0:
        return None
    return round(69 + 12 * np.log2(hz / 440))


def midi_to_name(midi):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{names[midi % 12]}{midi // 12 - 1}"


def nearest_chord_tone(midi_note, chord):
    """把一个音符映射到当前和弦里最近的和弦音"""
    candidates = [note + offset for note in chord for offset in [-12, 0, 12, 24]]
    return min(candidates, key=lambda x: abs(x - midi_note))


def get_chord_at_time(t):
    beat_dur = 60 / BPM
    chord_dur = beat_dur * BEATS_PER_CHORD
    progression_dur = chord_dur * len(CHORD_PROGRESSION)
    idx = int((t % progression_dur) / chord_dur) % len(CHORD_PROGRESSION)
    return CHORD_PROGRESSION[idx]


def detect_cat_segments(audio_clean, sr=SR):
    """用频率能量检测猫叫片段，接受已降噪的 audio 数组，返回 [(start, end, confidence), ...]"""

    # 猫叫主要频率范围：300～2000 Hz
    # 用 STFT 提取频谱，只看这个频段的能量
    hop = 512
    n_fft = 2048
    S = np.abs(librosa.stft(audio_clean, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    low, high = 300, 2000
    band = (freqs >= low) & (freqs <= high)
    band_energy = S[band, :].mean(axis=0)  # 每帧在该频段的平均能量

    # 自适应阈值：均值 + 0.8 倍标准差（保留能量峰值段）
    threshold = band_energy.mean() + 0.8 * band_energy.std()
    active = band_energy > threshold

    # 把连续激活帧合并成片段，合并间隔 < 0.3 秒的相邻段
    times = librosa.frames_to_time(np.arange(len(active)), sr=sr, hop_length=hop)
    MIN_SEG, MERGE_GAP = 0.1, 0.3

    segments, in_seg, seg_start = [], False, 0.0
    for i, (t, a) in enumerate(zip(times, active)):
        if a and not in_seg:
            in_seg, seg_start = True, t
        elif not a and in_seg:
            in_seg = False
            if t - seg_start >= MIN_SEG:
                segments.append([seg_start, t])

    if in_seg:
        if times[-1] - seg_start >= MIN_SEG:
            segments.append([seg_start, times[-1]])

    # 合并相邻片段
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    # 计算每段置信度（该段能量 / 阈值，>1 表示超过阈值）
    result = []
    for s, e in merged:
        s_frame = librosa.time_to_frames(s, sr=sr, hop_length=hop)
        e_frame = librosa.time_to_frames(e, sr=sr, hop_length=hop)
        conf = float(band_energy[s_frame:e_frame].mean() / threshold)
        result.append((s, e, min(conf, 1.0)))

    print(f"频率能量分析完成，检测阈值：{threshold:.4f}")

    # 每段前后各补 60ms，捕捉自然音头和尾音
    audio_dur = len(audio_clean) / sr
    PAD = 0.06
    padded = []
    for s, e in merged:
        padded.append((max(0.0, s - PAD), min(audio_dur, e + PAD)))
    return [(s, e, min(float(band_energy[
        librosa.time_to_frames(s, sr=sr, hop_length=hop):
        librosa.time_to_frames(e, sr=sr, hop_length=hop)
    ].mean() / threshold), 1.0)) for s, e in padded]


def extract_note_events(audio, sr, start, end):
    """从一段猫叫中提取多个音符事件（保留滑音轮廓）"""
    s, e = int(start * sr), int(end * sr)
    seg = audio[s:e]
    if len(seg) < 512:
        return []

    f0, voiced, _ = librosa.pyin(
        seg,
        fmin=librosa.note_to_hz('C3'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    times = librosa.times_like(f0, sr=sr, hop_length=512)

    events = []
    cur_pitch, cur_start = None, None
    SEMI_THRESH, MIN_DUR = 1.5, 0.15

    for t, pitch, v in zip(times, f0, voiced):
        if not v or pitch is None or np.isnan(pitch):
            if cur_pitch and (t - cur_start) >= MIN_DUR:
                m = hz_to_midi(cur_pitch)
                if m:
                    events.append({'start': start + cur_start, 'end': start + t,
                                   'midi': m, 'hz': cur_pitch})
            cur_pitch, cur_start = None, None
            continue
        if cur_pitch is None:
            cur_pitch, cur_start = pitch, t
        elif abs(hz_to_midi(pitch) - hz_to_midi(cur_pitch)) > SEMI_THRESH:
            if (t - cur_start) >= MIN_DUR:
                m = hz_to_midi(cur_pitch)
                if m:
                    events.append({'start': start + cur_start, 'end': start + t,
                                   'midi': m, 'hz': cur_pitch})
            cur_pitch, cur_start = pitch, t
        else:
            cur_pitch = (cur_pitch + pitch) / 2

    if cur_pitch and cur_start is not None:
        m = hz_to_midi(cur_pitch)
        if m:
            events.append({'start': start + cur_start, 'end': end,
                           'midi': m, 'hz': cur_pitch})
    return events


def _pitch_class_dist(midi, chord):
    """音高（按音级，忽略八度）到和弦任一和弦音的最小半音距离"""
    pc = midi % 12
    pcs = {c % 12 for c in chord}
    return min(min((pc - cpc) % 12, (cpc - pc) % 12) for cpc in pcs)


def _onset_offset(clip, sr):
    """片段内第一个真正发声（起音）的样本位置，用于对齐拍点"""
    if len(clip) == 0:
        return 0
    thr = (np.max(np.abs(clip)) + 1e-9) * 0.15
    above = np.abs(clip) > thr
    return int(np.argmax(above)) if above.any() else 0


# 每个风格的落点节奏型（4拍小节内的拍偏移）。
# 每项是 (乐句A, 乐句B)，奇偶小节交替 → 呼应/呼吸感；含切分(.5/.25/.75)与留白
RHYTHM_PATTERNS = {
    'jazz':      ([0, 1.5, 2.5, 3],  [0.5, 1, 2, 3.5]),   # swing 切分，密
    'folk':      ([0, 1, 2.5],       [0, 2, 3]),           # 中等，带一点弱拍
    'rb':        ([0, 1.75, 2.5],    [0.5, 2, 3.25]),      # 强切分
    'march':     ([0, 1, 2, 3],      [0, 1.5, 2]),         # 稳健 + 偶尔附点
    'lofi':      ([0, 2.5],          [1.5]),               # 稀疏慵懒
    'classical': ([0, 1, 2, 3],      [0, 1.5, 2.5]),       # 流动
}


def build_cat_track(sr, segments, note_events, total_samples, output_duration):
    """根据每段猫叫的自然音高智能排列：落点贴合当前和弦，并带上下起伏的旋律轮廓"""
    phrases = []
    for seg in segments:
        seg_start, seg_end = seg[0], seg[1]
        src_audio = seg[3] if len(seg) > 3 else None
        evs = sorted(
            [e for e in note_events if e['start'] >= seg_start - 0.1 and e['end'] <= seg_end + 0.1],
            key=lambda e: e['start']
        )
        if evs and src_audio is not None:
            s = int(seg_start * sr)
            e = int(seg_end * sr)
            clip = src_audio[s:min(e, len(src_audio))].copy()
            phrases.append({
                'duration': seg_end - seg_start,
                'pitch': evs[0]['midi'],          # 代表音高：取第一个音
                'clip': clip,
                'onset': _onset_offset(clip, sr),  # 真实起音位置
            })

    if not phrases:
        return np.zeros(total_samples)

    beat_dur = 60.0 / BPM
    BAR = 4  # 以 4 拍为一个节奏小节安排落点
    patterns = RHYTHM_PATTERNS.get(GENRE_KEY, ([0, 2], [0, 2.5]))

    # 按小节平铺节奏型，奇偶小节交替 → 乐句呼吸
    candidates = []
    b = 0
    while b * BAR * beat_dur < output_duration - 0.5:
        patt = patterns[b % len(patterns)]
        for off in patt:
            ct = (b * BAR + off) * beat_dur
            if ct < output_duration - 0.3:
                candidates.append(ct)
        b += 1
    candidates.sort()

    placed = []
    last_idx, last_pitch, want_up = -1, None, True
    next_free = 0.0

    for ct in candidates:
        if ct < next_free:
            continue  # 上一声还没放完 → 此处留白（休止）

        chord = get_chord_at_time(ct)

        # 1. 给每段打分：越贴近和弦音越好，紧跟上次的同一段加惩罚
        scored = []
        for idx, p in enumerate(phrases):
            dist = _pitch_class_dist(p['pitch'], chord)
            penalty = 3 if idx == last_idx else 0
            scored.append((dist + penalty, idx, p['pitch']))

        # 2. 取贴合度最好的一批候选
        best = min(s[0] for s in scored)
        cands = [s for s in scored if s[0] <= best + 1]

        # 3. 旋律轮廓：相对上一个音，交替往上 / 往下走
        if last_pitch is not None:
            directed = [c for c in cands
                        if (c[2] > last_pitch) == want_up and c[2] != last_pitch]
            chosen = directed or cands
        else:
            chosen = cands

        _, idx, pitch = random.choice(chosen)
        p = phrases[idx]
        placed.append((ct, p))

        last_idx, last_pitch = idx, pitch
        want_up = not want_up
        next_free = ct + p['duration'] * 0.7   # 留一点交叠余量，疏密随音长自然变化

    track = np.zeros(total_samples)

    for place_time, phrase in placed:
        seg = phrase['clip'].copy()
        if len(seg) < 512:
            continue

        fade = min(int(0.08 * sr), len(seg) // 3)
        seg[:fade]  *= np.hanning(fade * 2)[:fade]
        seg[-fade:] *= np.hanning(fade * 2)[fade:]

        # 按真实起音对齐拍点：起音落在 place_time，clip 的静音/淡入提前到拍点之前
        dest_s = int(place_time * sr) - phrase['onset']
        if dest_s < 0:
            seg = seg[-dest_s:]
            dest_s = 0
        dest_e = min(dest_s + len(seg), total_samples)
        track[dest_s:dest_e] += seg[:dest_e - dest_s]

    return track


def _chord_at(beat):
    ci = int(beat / BEATS_PER_CHORD) % len(CHORD_PROGRESSION)
    return CHORD_PROGRESSION[ci]


def _jazz_backing(midi, total_beats):
    """Jazz: swing 8分音符琶音，offbeat 延后产生摇摆感（单钢琴）"""
    midi.addProgramChange(0, 0, 0, 0)  # Acoustic Grand Piano
    SW = 0.17

    t = 0
    while t < total_beats:
        chord = _chord_at(t)
        root = chord[0]
        third = chord[1] if len(chord) > 1 else root + 4
        fifth = chord[2] if len(chord) > 2 else root + 7
        pattern = [
            (0,        root,      68), (0.5 + SW, third,    52),
            (1.0,      fifth,     58), (1.5 + SW, root+12,  50),
            (2.0,      root,      64), (2.5 + SW, third,    50),
            (3.0,      fifth,     56), (3.5 + SW, root+12,  48),
        ]
        for off, note, vel in pattern:
            if t + off < total_beats:
                midi.addNote(0, 0, note, t + off, 0.4, vel)
        t += BEATS_PER_CHORD


def _folk_backing(midi, total_beats):
    """Folk: 木吉他交替低音（1&3拍）+ 扫弦（2&4拍），单吉他"""
    midi.addProgramChange(0, 0, 0, 25)  # Steel Acoustic Guitar

    def strum(chord_notes, time, vel_base):
        for i, n in enumerate(chord_notes):
            if time + i * 0.02 < total_beats:
                midi.addNote(0, 0, n, time + i * 0.02, 0.5, max(32, vel_base - i * 5))

    t = 0
    while t < total_beats:
        chord = _chord_at(t)
        root = chord[0]
        fifth = chord[2] if len(chord) > 2 else root + 7
        bass_root = max(40, root - 12)
        bass_fifth = max(40, fifth - 12)

        if t < total_beats:
            midi.addNote(0, 0, bass_root, t, 0.72, 66)         # 1拍: 低音根音
        if t + 1.5 < total_beats:
            strum(chord, t + 1.5, 55)                           # 1.5拍: 扫弦
        if t + 2.5 < total_beats:
            strum(chord, t + 2.5, 46)                           # 2.5拍: 轻扫
        if t + 3 < total_beats:
            midi.addNote(0, 0, bass_fifth, t + 3, 0.72, 62)    # 3拍: 低音5度
        if t + 3.5 < total_beats:
            strum(chord, t + 3.5, 52)                           # 3.5拍: 扫弦
        if t + 4.5 < total_beats:
            strum(chord, t + 4.5, 44)                           # 4.5拍: 扫弦

        t += BEATS_PER_CHORD


def _rb_backing(midi, total_beats):
    """R&B: Rhodes 切分打点（1 / 1.75 / 2.5 / 3.75拍），单电钢"""
    midi.addProgramChange(0, 0, 0, 4)  # Electric Piano 1 (Rhodes)

    t = 0
    while t < total_beats:
        chord = _chord_at(t)
        for off, vel in [(0, 72), (1.75, 60), (2.5, 65), (3.75, 58)]:
            if t + off < total_beats:
                for i, n in enumerate(chord):
                    midi.addNote(0, 0, n, t + off, 0.4, max(38, vel - i * 6))
        t += BEATS_PER_CHORD


def _march_backing(midi, total_beats):
    """进行曲: 钢琴附点节奏（♩. ♪），强拍重音，单钢琴"""
    midi.addProgramChange(0, 0, 0, 0)  # Acoustic Grand Piano

    t = 0
    while t < total_beats:
        chord = _chord_at(t)
        for i, n in enumerate(chord):
            if t < total_beats:
                midi.addNote(0, 0, n, t, 1.45, max(48, 80 - i * 7))       # 附点4分
            if t + 1.5 < total_beats:
                midi.addNote(0, 0, n, t + 1.5, 0.42, max(38, 62 - i * 7)) # 8分
        t += BEATS_PER_CHORD


def _lofi_backing(midi, total_beats):
    """Lo-fi: 懒散 swing Rhodes 琶音，前半小节弹后半留白，单电钢"""
    midi.addProgramChange(0, 0, 0, 4)  # Electric Piano 1 (Rhodes)
    SW = 0.14

    t = 0
    while t < total_beats:
        chord = _chord_at(t)
        offsets = [0, 0.5 + SW, 1.0, 1.5 + SW]
        for i, n in enumerate(chord[:4]):
            if t + offsets[i] < total_beats:
                midi.addNote(0, 0, n, t + offsets[i], 0.45, 52 if i % 2 == 0 else 40)
        t += BEATS_PER_CHORD


def _classical_backing(midi, total_beats):
    """古典: 钢琴 Alberti 低音（root-5th-3rd-5th 16分音符组），单钢琴"""
    midi.addProgramChange(0, 0, 0, 0)  # Acoustic Grand Piano

    t = 0
    while t < total_beats:
        chord = _chord_at(t)
        root = chord[0]
        third = chord[1] if len(chord) > 1 else root + 4
        fifth = chord[2] if len(chord) > 2 else root + 7

        alberti = [root - 12, fifth - 12, third - 12, fifth - 12]
        for beat_off in range(min(BEATS_PER_CHORD, total_beats - t)):
            for j, note in enumerate(alberti):
                nt = t + beat_off + j * 0.25
                if nt < total_beats:
                    midi.addNote(0, 0, max(28, note), nt, 0.22, 62 if j == 0 else 44)

        t += BEATS_PER_CHORD


def generate_backing(duration, sr):
    """风格差异化伴奏：每种风格有独立节奏型、乐器编配、演奏方式"""
    midi = MIDIFile(1)
    midi.addTempo(0, 0, BPM)
    total_beats = int(duration * BPM / 60) + 8

    backing_fns = {
        'jazz':      _jazz_backing,
        'folk':      _folk_backing,
        'rb':        _rb_backing,
        'march':     _march_backing,
        'lofi':      _lofi_backing,
        'classical': _classical_backing,
    }
    backing_fns.get(GENRE_KEY, _lofi_backing)(midi, total_beats)

    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        midi_path = f.name
        midi.writeFile(f)

    wav_path = midi_path.replace('.mid', '.wav')
    subprocess.run(
        ['fluidsynth', '-ni', '-F', wav_path, '-r', str(sr), SOUNDFONT, midi_path],
        capture_output=True
    )
    os.unlink(midi_path)

    if not os.path.exists(wav_path):
        print("⚠️  FluidSynth 渲染失败，跳过伴奏")
        return np.zeros(int(duration * sr))

    backing, _ = librosa.load(wav_path, sr=sr, mono=True)
    os.unlink(wav_path)
    return backing


def analyze_files(input_paths):
    """加载 + 降噪 + 检测猫叫 + 提取音高，返回 (segments, recommendations)。
    segments: [{id, start, end, confidence, pitch, events, audio}, ...]（audio 为降噪后整段）"""
    import noisereduce as nr
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    segments, sid = [], 0
    for path in input_paths:
        audio, _ = librosa.load(path, sr=SR, mono=True)
        noise_sample = audio[:int(SR * 1.0)]
        audio_clean = nr.reduce_noise(y=audio, sr=SR, y_noise=noise_sample, stationary=True)
        for s, e, c in detect_cat_segments(audio_clean):
            evs = extract_note_events(audio_clean, SR, s, e)
            if not evs:
                continue
            segments.append({
                'id': sid, 'start': float(s), 'end': float(e), 'confidence': float(c),
                'pitch': evs[0]['midi'], 'events': evs, 'audio': audio_clean,
            })
            sid += 1
        print(f"降噪完成")

    recs = []
    if segments:
        recs = recommend_genres([(x['start'], x['end'], x['confidence']) for x in segments])
    return segments, recs


def render_from_segments(segments, genre_key, out_path):
    """用（已筛选的）segments 渲染一首 MP3 到 out_path，返回时长（秒）"""
    global CHORD_PROGRESSION
    apply_genre(genre_key)

    all_segments = [(x['start'], x['end'], x['confidence'], x['audio']) for x in segments]
    all_events = []
    for x in segments:
        all_events.extend(x['events'])

    CHORD_PROGRESSION = build_generated_progression(genre_key, all_events)

    max_end = max(x['end'] for x in segments)
    out_dur = max(OUTPUT_DURATION, max_end + 3)
    out_samples = int(out_dur * SR)

    cat_track = build_cat_track(SR, all_segments, all_events, out_samples, out_dur)

    backing = generate_backing(out_dur, SR)
    if len(backing) < out_samples:
        backing = np.pad(backing, (0, out_samples - len(backing)))
    else:
        backing = backing[:out_samples]

    eps = 1e-8
    cat_norm = cat_track / (np.max(np.abs(cat_track)) + eps) * 0.75
    back_norm = backing / (np.max(np.abs(backing)) + eps) * 0.4
    mixed = cat_norm + back_norm
    mixed = mixed / (np.max(np.abs(mixed)) + eps) * 0.9

    wav_path = out_path.replace('.mp3', '.wav')
    sf.write(wav_path, mixed, SR)
    subprocess.run(
        ['ffmpeg', '-y', '-i', wav_path, '-codec:a', 'libmp3lame', '-b:a', '192k', out_path],
        capture_output=True, text=True
    )
    if os.path.exists(wav_path):
        os.unlink(wav_path)
    return out_dur


def generate(input_paths, genre_key=None):
    """CLI 入口：分析 + 渲染，输出 output_<genre>.mp3"""
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    print(f"\n{'='*50}")
    print(f"素材：{', '.join(os.path.basename(p) for p in input_paths)}")
    print(f"{'='*50}\n")
    print("分析中（加载 / 降噪 / 检测 / 提取音高）...")

    segments, recommendations = analyze_files(input_paths)
    if not segments:
        print("❌ 所有素材中均未检测到可用猫叫，请换一段更清晰的录音")
        return

    print(f"\n✅ 共 {len(segments)} 段可用猫叫：")
    for x in segments:
        print(f"   #{x['id']}  {x['start']:.2f}s-{x['end']:.2f}s  "
              f"{midi_to_name(x['pitch'])}（置信度 {x['confidence']:.2f}）")

    print(f"\n🎵 风格推荐（按适配度排序）：")
    for rank, (k, score) in enumerate(recommendations):
        print(f"   {rank+1}. {GENRES[k]['label']}  {int(score*100)}%")

    if genre_key is None:
        genre_key = recommendations[0][0]
    print(f"\n→ 使用风格：{GENRES[genre_key]['label']}（BPM {GENRES[genre_key]['bpm']}）")

    output_dir = os.path.dirname(os.path.abspath(input_paths[0]))
    out_path = os.path.join(output_dir, f'output_{genre_key}.mp3')
    render_from_segments(segments, genre_key, out_path)

    if os.path.exists(out_path):
        print(f"\n✅ 完成！输出：{out_path}")
    else:
        print(f"❌ 输出失败")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        keys = ' | '.join(GENRES.keys())
        print(f"用法：python generate.py <录音1> [录音2 ...] [风格]")
        print(f"风格可选：{keys}（不填则自动推荐最适合的）")
        print(f"例如：python generate.py ~/Desktop/cat.mp3 ~/Desktop/cat2.mp3 jazz")
        sys.exit(1)
    args = sys.argv[1:]
    genre = args[-1] if args[-1] in GENRES else None
    paths = [a for a in args if a not in GENRES]
    generate(paths, genre)
