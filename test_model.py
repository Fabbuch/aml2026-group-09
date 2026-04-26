"""
Tests for PoPEViT.

Run with:  python test_model.py
"""

import torch
import torch.nn.functional as F
from model import PoPEViT

NUM_CLASSES = 4
IMAGE_SIZE = 256
BATCH_SIZE = 2


def make_model(**kwargs):
    defaults = dict(image_size=IMAGE_SIZE, patch_size=16, num_classes=NUM_CLASSES,
                    dim=128, depth=2, heads=4, mlp_dim=256, dropout=0.1)
    defaults.update(kwargs)
    return PoPEViT(**defaults)


def test_output_shape():
    model = make_model()
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES), f"Expected {(BATCH_SIZE, NUM_CLASSES)}, got {out.shape}"
    print("PASS  output shape")


def test_output_is_logits():
    # Output should be raw logits, not probabilities — no value should be forced to [0,1]
    model = make_model()
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(x)
    probs = F.softmax(out, dim=-1)
    assert probs.shape == (BATCH_SIZE, NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5)
    print("PASS  output is valid logit distribution after softmax")


def test_backward_pass():
    # Gradients must flow through the full model including PoPE layers
    model = make_model()
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    loss = F.cross_entropy(model(x), labels)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("PASS  backward pass / gradients flow to all parameters")


def test_different_patch_sizes():
    for patch_size in [8, 16, 32]:
        model = make_model(patch_size=patch_size)
        out = model(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert out.shape == (1, NUM_CLASSES)
    print("PASS  patch sizes 8 / 16 / 32")


def test_batch_size_one():
    # BatchNorm-free architecture should handle batch size 1 fine
    model = make_model()
    out = model(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (1, NUM_CLASSES)
    print("PASS  batch size 1")


def test_dataloader_compatibility():
    """
    Simulates a single DataLoader batch without requiring the actual dataset on disk.
    Images are (3, 256, 256) float32 in [0, 1] — matching BrainTumorDataset output
    after v2.ToDtype(torch.float32, scale=True).
    """
    model = make_model()
    images = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)   # float32, [0, 1]
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    out = model(images)
    loss = F.cross_entropy(out, labels)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)
    assert loss.item() > 0
    print("PASS  dataloader-format batch (float32, [0,1])")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_output_shape()
    test_output_is_logits()
    test_backward_pass()
    test_different_patch_sizes()
    test_batch_size_one()
    test_dataloader_compatibility()
    print("\nAll tests passed.")
