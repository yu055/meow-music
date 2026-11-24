#!/usr/bin/env python3
"""诊断脚本：可视化 YAMNet 对每一帧的猫叫评分"""
import sys
import numpy as np
import librosa
import noisereduce as nr

def diagnose(audio_path):
    import tensorflow_hub as hub

    print("加载音频...")
    audio16k, _ = librosa.load(audio_path, sr=16000, mono=True)
    noise_sample = audio16k[:int(16000 * 1.0)]
    audio16k = nr.reduce_noise(y=audio16k, sr=16000, y_noise=noise_sample, stationary=True)

    print("加载 YAMNet...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    scores, _, _ = model(audio16k)
    scores = scores.numpy()

    CAT_CLASS = 81
    RELATED = [78, 80, 81, 82, 83]  # Animal / Domestic / Cat / Kitten / Purring
    FRAME_DUR = 0.48

    cat_scores = scores[:, CAT_CLASS]
    combined = scores[:, RELATED].max(axis=1)  # 相关类别最高分

    print(f"\n最高猫叫置信度（Cat类）：{cat_scores.max():.3f}")
    print(f"最高置信度（Animal相关类）：{combined.max():.3f}")
    print(f"\n每帧评分（Cat类 | 相关类 | 时间）：")
    print("-" * 45)

    for i, (cat, comb) in enumerate(zip(cat_scores, combined)):
        t = i * FRAME_DUR
        bar_cat = "█" * int(cat * 30)
        bar_comb = "░" * int(comb * 30)
        marker = " ← 猫叫" if cat > 0.05 else ("← 相关" if comb > 0.1 else "")
        print(f"{t:5.1f}s  Cat:{cat:.2f} {bar_cat:<10}  Combined:{comb:.2f} {marker}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法：python diagnose.py <音频路径>")
        sys.exit(1)
    diagnose(sys.argv[1])
