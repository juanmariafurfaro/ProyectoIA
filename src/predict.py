import argparse
import torch
import torch.nn.functional as F

from .utils.video_io import sample_clip
from .data.video_transforms import ClipTransform
from .models.cnn_lstm import CNNLSTM


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--video', type=str, required=True)
    ap.add_argument('--num_frames', type=int, default=16)
    return ap.parse_args()


def main():
    args = get_args()
    ckpt = torch.load(args.ckpt, map_location='cpu')

    classes = ckpt.get('classes', ['incorrect','correct'])
    model_args = ckpt.get('args', {})
    model = CNNLSTM(num_classes=len(classes),
                    backbone=model_args.get('backbone', 'resnet18'),
                    hidden_size=model_args.get('hidden_size', 256),
                    bidirectional=model_args.get('bidirectional', False))
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    clip = sample_clip(args.video, args.num_frames)
    t = ClipTransform((224,224), train=False)
    clip_t = t(clip).unsqueeze(0)  # [1,T,C,H,W]

    with torch.no_grad():
        logits = model(clip_t)
        probs = F.softmax(logits, dim=1).squeeze(0)
        conf, pred = torch.max(probs, dim=0)

    print(f"Predicción: {classes[pred.item()]} (conf={conf.item():.3f})")


if __name__ == '__main__':
    main()