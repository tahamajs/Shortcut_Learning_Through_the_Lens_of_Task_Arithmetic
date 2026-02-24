import pytest

torch = pytest.importorskip("torch")

from shortcut_patcher.src.edit_model import apply_task_edit


def test_apply_task_edit_negation():
    base = {"w": torch.tensor([1.0, 2.0])}
    vec = {"w": torch.tensor([0.25, -0.5])}
    edited = apply_task_edit(base, [vec], alphas=[-1.0])
    assert torch.allclose(edited["w"], torch.tensor([0.75, 2.5]))


def test_apply_task_edit_composition():
    base = {"w": torch.tensor([1.0, 2.0])}
    vec1 = {"w": torch.tensor([0.5, 0.5])}
    vec2 = {"w": torch.tensor([-0.5, 1.0])}
    edited = apply_task_edit(base, [vec1, vec2], alphas=[1.0, -0.5])
    assert torch.allclose(edited["w"], torch.tensor([1.75, 2.0]))
