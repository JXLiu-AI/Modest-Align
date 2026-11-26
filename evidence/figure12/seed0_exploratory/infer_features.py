"""Classify authorized frozen features with the released adapter checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adapter_model import load_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=Path('model.safetensors'))
    parser.add_argument('--features', type=Path, required=True,
                        help='NPZ with image_features [N,1536], text_features [C,1024], optional targets [N]')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = np.load(args.features, allow_pickle=False)
    visual = torch.tensor(data['image_features'], dtype=torch.float32)
    text = torch.tensor(data['text_features'], dtype=torch.float32)
    assert visual.ndim == text.ndim == 2
    assert visual.shape[1] == 1536 and text.shape[1] == 1024
    assert visual.shape[0] > 0 and text.shape[0] > 1
    assert torch.isfinite(visual).all() and torch.isfinite(text).all()
    model = load_adapter(args.weights)
    with torch.inference_mode():
        scores = model(visual, text)
    assert torch.isfinite(scores).all()
    prediction = scores.argmax(1).tolist()
    result = {'n': len(prediction), 'predictions': prediction,
              'cosine_scores': scores.tolist(), 'training_seed': 0,
              'scope': 'One COCO-trained checkpoint; no test-time perturbation or fitting'}
    if 'targets' in data:
        target = data['targets'].tolist()
        assert len(target) == len(prediction)
        assert all(isinstance(t, int) and 0 <= t < len(text) for t in target)
        assert set(target) == set(range(len(text))), 'Balanced accuracy requires every class'
        result['top1_percent'] = 100 * sum(a == b for a, b in zip(target, prediction)) / len(target)
        recalls = [sum(t == p == c for t, p in zip(target, prediction)) / target.count(c)
                   for c in range(len(text))]
        result['balanced_accuracy_percent'] = 100 * sum(recalls) / len(recalls)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
