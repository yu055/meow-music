#!/usr/bin/env python3
"""
猫叫交响曲 - 本地开发后端（FastAPI，同步版）
流程：/upload → /analyze → （前端审核音符、选风格）→ /generate → 播放
会话状态存内存，适合本地联调；上线换 Celery + Redis + 对象存储。
"""

import os
import uuid
import shutil
import tempfile

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import generate as gen

app = FastAPI(title="Meow Symphony API")

# 工作目录：上传原始文件、每段预览、成品 MP3
WORK_DIR = os.path.join(tempfile.gettempdir(), "meow_sessions")
os.makedirs(WORK_DIR, exist_ok=True)

# 内存会话表：session_id -> {segments, uploads, previews, genre_label...}
SESSIONS = {}


def _session_dir(sid):
    d = os.path.join(WORK_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return d


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    """上传一个或多个音频文件，返回 session_id"""
    sid = uuid.uuid4().hex[:12]
    sdir = _session_dir(sid)
    saved = []
    for f in files:
        dest = os.path.join(sdir, os.path.basename(f.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    SESSIONS[sid] = {"uploads": saved}
    return {"session_id": sid, "files": [os.path.basename(p) for p in saved]}


@app.post("/analyze/{sid}")
def analyze(sid: str):
    """降噪 + 检测猫叫 + 提取音高，切出每段预览音频，返回片段列表 + 风格推荐"""
    sess = SESSIONS.get(sid)
    if not sess:
        raise HTTPException(404, "session 不存在，请先 /upload")

    segments, recs = gen.analyze_files(sess["uploads"])
    if not segments:
        raise HTTPException(422, "未检测到可用猫叫，请换一段更清晰的录音")

    sdir = _session_dir(sid)
    seg_out = []
    for x in segments:
        # 切出该段预览 WAV（含检测时的 60ms padding 已在 start/end 内）
        s, e = int(x["start"] * gen.SR), int(x["end"] * gen.SR)
        clip = x["audio"][s:min(e, len(x["audio"]))]
        wav_name = f"seg_{x['id']}.wav"
        sf.write(os.path.join(sdir, wav_name), clip, gen.SR)
        seg_out.append({
            "id": x["id"],
            "start": round(x["start"], 2),
            "end": round(x["end"], 2),
            "confidence": round(x["confidence"], 2),
            "pitch": x["pitch"],                       # MIDI 音高（取第一个音）
            "pitch_name": gen.midi_to_name(x["pitch"]),
            "audio_url": f"/media/{sid}/{wav_name}",
        })

    sess["segments"] = segments      # 保留完整对象（含 audio/events）供渲染
    recommendations = [
        {"key": k, "label": gen.GENRES[k]["label"],
         "bpm": gen.GENRES[k]["bpm"], "fit": round(score, 2)}
        for k, score in recs
    ]
    return {"segments": seg_out, "recommendations": recommendations}


@app.post("/generate/{sid}")
def generate(sid: str, genre: str, keep: str = ""):
    """按选定风格生成成品。keep：保留的片段 id（逗号分隔），空=全部"""
    sess = SESSIONS.get(sid)
    if not sess or "segments" not in sess:
        raise HTTPException(404, "请先 /analyze")
    if genre not in gen.GENRES:
        raise HTTPException(400, f"未知风格 {genre}")

    segments = sess["segments"]
    if keep.strip():
        keep_ids = {int(i) for i in keep.split(",") if i.strip().isdigit()}
        segments = [x for x in segments if x["id"] in keep_ids]
    if not segments:
        raise HTTPException(422, "没有保留任何片段")

    sdir = _session_dir(sid)
    out_name = f"output_{genre}.mp3"
    out_path = os.path.join(sdir, out_name)
    duration = gen.render_from_segments(segments, genre, out_path)

    if not os.path.exists(out_path):
        raise HTTPException(500, "生成失败")
    return {
        "mp3_url": f"/media/{sid}/{out_name}",
        "duration": round(duration, 1),
        "genre": genre,
        "genre_label": gen.GENRES[genre]["label"],
    }


@app.get("/media/{sid}/{name}")
def media(sid: str, name: str):
    """提供会话内的音频文件（预览片段 / 成品）"""
    path = os.path.join(WORK_DIR, sid, os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "文件不存在")
    return FileResponse(path)


# 前端静态页面（/ → frontend/index.html）
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
