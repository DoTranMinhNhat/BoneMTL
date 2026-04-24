# tests/test_model.py
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import BoneMTL

model = BoneMTL(num_tumor_types=9, pretrained=False)
model.eval()


def test_output_keys():
    x   = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert set(out.keys()) == {'tier1', 'tier2', 'tier3', 'mask'}


def test_output_shapes():
    x   = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out['tier1'].shape == torch.Size([2, 1])
    assert out['tier2'].shape == torch.Size([2, 1])
    assert out['tier3'].shape == torch.Size([2, 9])
    assert out['mask'].shape  == torch.Size([2, 1, 256, 256])


def test_output_dtype():
    x   = torch.randn(2, 3, 256, 256)
    out = model(x)
    for k, v in out.items():
        assert v.dtype == torch.float32, f"{k} dtype sai"


def test_no_nan():
    x   = torch.randn(2, 3, 256, 256)
    out = model(x)
    for k, v in out.items():
        assert not torch.isnan(v).any(), f"{k} có NaN"


def test_batch_size_1():
    x   = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out['tier1'].shape == torch.Size([1, 1])
    assert out['mask'].shape  == torch.Size([1, 1, 256, 256])


def test_parameter_count():
    total = sum(p.numel() for p in model.parameters())
    assert total > 30_000_000


if __name__ == '__main__':
    tests = [
        test_output_keys,
        test_output_shapes,
        test_output_dtype,
        test_no_nan,
        test_batch_size_1,
        test_parameter_count,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)}")