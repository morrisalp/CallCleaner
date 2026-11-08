"""CallCleaner: denoise animal vocalization recordings (no GPU, detector or text prompt needed).

    python denoise.py <input file or folder> <output folder> [--keep-sr] [--device cpu|cuda|mps]

Input audio is downmixed to mono and resampled to 16 kHz (the model's rate); long files are processed in
overlapping chunks and cross-faded. Output: <output folder>/<name>.wav at 16 kHz, or at the input's rate with
--keep-sr (content above 8 kHz is not restored). Weights are downloaded from huggingface.co/malper/CallCleaner.
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from biodenoising.denoiser.pretrained import biodenoising16k_dns48
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

SR = 16000
AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("output")
ap.add_argument("--checkpoint", default=None, help="local .safetensors file (default: download from Hugging Face)")
ap.add_argument("--keep-sr", action="store_true", help="resample the output back to the input's sample rate")
ap.add_argument("--chunk-s", type=float, default=60.0)
ap.add_argument("--overlap-s", type=float, default=1.0)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

model = biodenoising16k_dns48(pretrained=False)
model.load_state_dict(load_file(args.checkpoint or hf_hub_download("malper/CallCleaner", "model.safetensors")))
model.eval().to(args.device)


def denoise(y):
    """Chunked inference with a linear cross-fade over the overlap."""
    chunk, hop = int(args.chunk_s * SR), int((args.chunk_s - args.overlap_s) * SR)
    if len(y) <= chunk:
        starts = [0]
    else:
        starts = list(range(0, len(y) - int(args.overlap_s * SR), hop))
    out, weight = np.zeros(len(y), np.float32), np.zeros(len(y), np.float32)
    fade = int(args.overlap_s * SR)
    for s in starts:
        seg = y[s : s + chunk]
        with torch.no_grad():
            est = model(torch.tensor(seg, device=args.device)[None, None])[0, 0, : len(seg)].cpu().numpy()
        w = np.ones(len(seg), np.float32)
        if s > 0:
            w[:fade] = np.linspace(0, 1, fade, dtype=np.float32)[: len(seg)]
        if s + chunk < len(y):
            w[-fade:] = np.minimum(w[-fade:], np.linspace(1, 0, fade, dtype=np.float32))
        out[s : s + len(seg)] += est * w
        weight[s : s + len(seg)] += w
    return out / np.maximum(weight, 1e-6)


src = Path(args.input)
files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in AUDIO)
dest = Path(args.output)
dest.mkdir(parents=True, exist_ok=True)
for f in tqdm(files, desc="denoise"):
    y, sr_in = librosa.load(f, sr=None, mono=True)
    est = denoise(librosa.resample(y, orig_sr=sr_in, target_sr=SR) if sr_in != SR else y)
    sr_out = SR
    if args.keep_sr and sr_in != SR:
        est, sr_out = librosa.resample(est, orig_sr=SR, target_sr=sr_in), sr_in
    rel = f.relative_to(src).with_suffix(".wav") if src.is_dir() else Path(f.stem + ".wav")
    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest / rel, est, sr_out)
print(f"{len(files)} file(s) -> {dest}")
