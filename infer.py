import os
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from models import build_seg_model
from clip_tokens import CLIPTeacher


def load_image(path, size=512):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size))
    t = transforms.ToTensor()
    return t(img).unsqueeze(0)


def colorize(mask, num_classes):
    rng = np.random.RandomState(42)
    palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_classes):
        out[mask == i] = palette[i]
    return Image.fromarray(out)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='checkpoints/best.pth')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--out', type=str, default='pred.png')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_seg_model(cfg['model'], num_seen_classes=cfg['data']['num_seen']).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'], strict=False)
    model.eval()

    x = load_image(args.image, size=cfg['data'].get('img_size', 512)).to(device)

    with torch.no_grad():
        out = model(x)
        logits = out['logits']
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    colorize(pred, cfg['data']['num_seen']).save(args.out)
    print('Saved', args.out)


if __name__ == '__main__':
    main()