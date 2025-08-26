import cv2
import numpy as np
from typing import List
import random

SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _read_all_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Video vacío o no legible: {path}")
    return frames


def uniform_sample_indices(n_frames: int, T: int) -> List[int]:
    if T <= 1:
        return [min(n_frames - 1, 0)]
    idx = np.linspace(0, n_frames - 1, T)
    return idx.round().astype(int).tolist()


def sample_clip(path: str, num_frames: int) -> List[np.ndarray]:
    frames = _read_all_frames(path)
    idxs = uniform_sample_indices(len(frames), num_frames)
    return [frames[i] for i in idxs]


def sample_clip_random(path: str, num_frames: int, n_total: int | None = None):
    frames = _read_all_frames(path)
    N = len(frames) if n_total is None else min(len(frames), n_total)
    if N <= num_frames:
        idxs = uniform_sample_indices(N, num_frames)
        return [frames[i] for i in idxs]
    # ventana aleatoria
    start = random.randint(0, N - num_frames)
    return frames[start:start+num_frames]
