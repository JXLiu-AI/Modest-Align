"""Extract private authorized features; never uploads data or downloads encoders."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True,
                        help='Private JSON: images, class_prompts, optional targets')
    parser.add_argument('--dino-repo', type=Path, required=True,
                        help='Previously obtained official DINOv2 source checkout')
    parser.add_argument('--dino-weights', type=Path, required=True)
    parser.add_argument('--bge-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = json.loads(args.manifest.read_text())
    assert data['images'] and len(data['class_prompts']) >= 2
    assert len(set(data['class_prompts'])) == len(data['class_prompts'])
    tokenizer = AutoTokenizer.from_pretrained(args.bge_dir, local_files_only=True)
    bge = AutoModel.from_pretrained(args.bge_dir, local_files_only=True).float().eval()
    prompts = [re.sub(r'\s{2,}', ' ', re.sub(r'([.!"()*#:;~])', ' ', s.lower())).strip()
               for s in data['class_prompts']]
    with torch.inference_mode():
        tokens = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        text = bge(**tokens)[0][:, 0].float().numpy()
    del bge
    dino = torch.hub.load(str(args.dino_repo), 'dinov2_vitg14', source='local', pretrained=False)
    try:
        weights = torch.load(args.dino_weights, map_location='cpu', weights_only=True)
    except TypeError:
        raise RuntimeError('Use a PyTorch version supporting weights_only=True for encoder weight loading')
    dino.load_state_dict(weights, strict=True)
    dino = dino.float().eval()
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    visual = []
    with torch.inference_mode():
        for value in data['images']:
            path = Path(value)
            if not path.is_absolute():
                path = args.manifest.parent / path
            with Image.open(path) as image:
                visual.append(dino(transform(image.convert('RGB')).unsqueeze(0)).float().numpy()[0])
    arrays = {'image_features': np.stack(visual), 'text_features': text}
    if 'targets' in data:
        assert len(data['targets']) == len(visual)
        arrays['targets'] = np.asarray(data['targets'], dtype=np.int64)
    assert arrays['image_features'].shape == (len(visual), 1536)
    assert text.shape == (len(prompts), 1024)
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez(args.output, **arrays)


if __name__ == '__main__':
    main()
