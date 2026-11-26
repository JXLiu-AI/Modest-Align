"""Check all released tensors and a non-medical deterministic inference fixture."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from adapter_model import load_adapter


def main():
    root = Path(__file__).resolve().parent
    reference = json.loads((root / 'model_verification.json').read_text())
    torch.set_num_threads(2)
    model = load_adapter(root / 'model.safetensors')
    fingerprint = hashlib.sha256()
    state = model.state_dict()
    assert len(state) == 57
    for name, tensor in sorted(state.items()):
        assert torch.isfinite(tensor).all()
        fingerprint.update(name.encode())
        fingerprint.update(str(tensor.dtype).encode())
        fingerprint.update(str(tuple(tensor.shape)).encode())
        fingerprint.update(tensor.contiguous().numpy().tobytes())
    assert fingerprint.hexdigest() == reference['tensor_fingerprint']
    with safe_open(root / 'model.safetensors', framework='pt', device='cpu') as file:
        assert file.metadata()['source_checkpoint_sha256'] == reference['source_checkpoint_sha256']
    with torch.inference_mode():
        visual = ((torch.arange(3 * 1536).reshape(3, 1536) % 97).float() - 48) / 48
        text = ((torch.arange(4 * 1024).reshape(4, 1024) % 89).float() - 44) / 44
        scores = model(visual, text)
    delta = (scores - torch.tensor(reference['golden_synthetic_scores'])).abs().max().item()
    assert delta < 1e-5, delta
    print(json.dumps({'status': 'PASS', 'tensor_count': len(state),
                      'tensor_fingerprint': fingerprint.hexdigest(),
                      'synthetic_fixture_max_score_error': delta}, indent=2))


if __name__ == '__main__':
    main()
