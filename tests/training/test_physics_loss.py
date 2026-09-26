import torch
from torch import nn

from pfem_cpfe.training.loss import relative_equilibrium_loss


def test_physics_loss_uses_no_solution_labels(unlabeled_case, model):
    assert "u_exact" not in unlabeled_case

    loss, metrics = relative_equilibrium_loss(model, unlabeled_case)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert set(metrics) == {
        "relative_residual",
        "state_penalty",
        "invalid_fraction",
    }
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_exact_equilibrium_field_has_near_zero_loss(
    equilibrated_case, identity_predictor
):
    loss, _ = relative_equilibrium_loss(identity_predictor, equilibrated_case)

    assert loss.item() < 1.0e-10


def test_invalid_prediction_receives_finite_penalty(unlabeled_case):
    class InvalidPredictor(nn.Module):
        def forward(self, xyz, features, mask):
            output = torch.zeros(
                (*xyz.shape[:2], 3), dtype=xyz.dtype, device=xyz.device
            )
            scale = torch.linspace(0.0, -3.0, xyz.shape[1], dtype=xyz.dtype)
            return output + scale[None, :, None]

    loss, metrics = relative_equilibrium_loss(InvalidPredictor(), unlabeled_case)

    assert torch.isfinite(loss)
    assert metrics["state_penalty"] > 0.0 or metrics["invalid_fraction"] > 0.0
