# src/predict.py — acepta archivo o carpeta
import argparse
import os
import torch
import torch.nn.functional as F

from .utils.video_io import sample_clip, SUPPORTED_EXTS
from .data.video_transforms import ClipTransform
from .models.cnn_lstm import CNNLSTM


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='Ruta al checkpoint .pt')
    ap.add_argument('--video', type=str, required=True, help='Ruta a un video o a una carpeta con videos')
    ap.add_argument('--num_frames', type=int, default=16)
    return ap.parse_args()


def load_model(ckpt_path: str) -> tuple[CNNLSTM, list[str]]:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    classes = ckpt.get('classes', ['incorrect', 'correct'])
    model_args = ckpt.get('args', {})
    model = CNNLSTM(
        num_classes=len(classes),
        backbone=model_args.get('backbone', 'resnet18'),
        hidden_size=model_args.get('hidden_size', 256),
        bidirectional=model_args.get('bidirectional', False)
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, classes


#def predict_one(model, classes, video_path, num_frames, k=5):
#    import torch.nn.functional as F
#    from .data.video_transforms import ClipTransform
#    from .utils.video_io import sample_clip_random
#
#    t = ClipTransform((224,224), train=False)
#    votes = torch.zeros(len(classes), dtype=torch.float32)
#
#    with torch.no_grad():
#        for _ in range(k):
#            clip = sample_clip_random(video_path, num_frames)
#            clip_t = t(clip).unsqueeze(0)
#            logits = model(clip_t)
#            probs = F.softmax(logits, dim=1).squeeze(0)
#            votes += probs
#
#    conf, pred = torch.max(votes / k, dim=0)
#    return classes[pred.item()], float(conf.item())

def predict_one(model, classes, video_path, num_frames, k=5, threshold=0.70):
    import torch
    import torch.nn.functional as F
    from .data.video_transforms import ClipTransform
    from .utils.video_io import _read_all_frames

    frames = _read_all_frames(video_path)
    T = len(frames)
    if T < num_frames:
        frames = frames + [frames[-1]] * (num_frames - T)
        T = len(frames)

    # K ventanas solapadas a lo largo del video
    step = max(1, (T - num_frames) // max(1, k - 1))
    starts = [i for i in range(0, max(1, T - num_frames + 1), step)][:k] or [0]

    t = ClipTransform((224,224), train=False)
    votes = torch.zeros(len(classes), dtype=torch.float32)

    with torch.no_grad():
        for s in starts:
            clip = frames[s:s+num_frames]
            if len(clip) < num_frames:
                clip = clip + [frames[-1]] * (num_frames - len(clip))
            clip_t = t(clip).unsqueeze(0)
            logits = model(clip_t)
            probs = F.softmax(logits, dim=1).squeeze(0)
            votes += probs

    probs_mean = votes / len(starts)
    conf, pred = torch.max(probs_mean, dim=0)
    label = classes[pred.item()]
    # política conservadora: si hay poca confianza, inclinamos a "incorrect"
    if conf.item() < threshold and "incorrect" in classes:
        label = "incorrect"
    return label, float(conf.item())




def iter_video_files(root: str):
    if os.path.isfile(root):
        yield root
        return
    # recorrer carpeta (y subcarpetas) y devolver videos
    exts = {e.lower() for e in SUPPORTED_EXTS}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            _, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                yield os.path.join(dirpath, fn)


def main():
    args = get_args()
    model, classes = load_model(args.ckpt)

    paths = list(iter_video_files(args.video))
    if not paths:
        print(f"No se encontraron videos en: {args.video}")
        return

    for p in paths:
        try:
            label, conf = predict_one(model, classes, p, args.num_frames)
            print(f"[OK] {p} -> {label} (conf={conf:.3f})")
        except Exception as e:
            print(f"[ERR] {p} -> {e}")


if __name__ == '__main__':
    main()
