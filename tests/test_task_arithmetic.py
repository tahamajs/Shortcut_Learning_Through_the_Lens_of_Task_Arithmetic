import torch

from shortcut_patcher.src.edit_model import apply_task_edit


def test_apply_task_edit_negation():
    base = {"w": torch.tensor([1.0, 2.0])}
    vec = {"w": torch.tensor([0.25, -0.5])}
    edited = apply_task_edit(base, vec, alpha=-1.0)
    assert torch.allclose(edited["w"], torch.tensor([0.75, 2.5]))
