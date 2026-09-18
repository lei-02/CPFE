import pytest
import torch

from pfem_cpfe.operator.transolver import TransolverCPFE


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_transolver_outputs_only_nodal_displacement(node_batch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = TransolverCPFE(
        feature_dim=node_batch.features.shape[-1],
        hidden=64,
        layers=2,
        heads=4,
        slices=16,
    ).to(device)

    output = model(
        node_batch.xyz.to(device),
        node_batch.features.to(device),
        node_batch.mask.to(device),
    )

    assert output.shape == (*node_batch.xyz.shape[:2], 3)
    assert output.device.type == device
    torch.testing.assert_close(
        output[~node_batch.mask.to(device)],
        torch.zeros_like(output[~node_batch.mask.to(device)]),
    )


def test_node_permutation_equivariance(node_batch):
    torch.manual_seed(23)
    model = TransolverCPFE(
        feature_dim=node_batch.features.shape[-1],
        hidden=32,
        layers=2,
        heads=4,
        slices=8,
    ).eval()
    permutation = torch.randperm(node_batch.xyz.shape[1])
    inverse = torch.argsort(permutation)

    reference = model(node_batch.xyz, node_batch.features, node_batch.mask)
    permuted = model(
        node_batch.xyz[:, permutation],
        node_batch.features[:, permutation],
        node_batch.mask[:, permutation],
    )

    torch.testing.assert_close(permuted[:, inverse], reference, rtol=2.0e-5, atol=2.0e-6)


def test_transolver_backpropagates_through_valid_nodes(node_batch):
    model = TransolverCPFE(
        feature_dim=node_batch.features.shape[-1],
        hidden=32,
        layers=1,
        heads=4,
        slices=8,
    )
    features = node_batch.features.clone().requires_grad_(True)

    loss = model(node_batch.xyz, features, node_batch.mask).square().sum()
    loss.backward()

    assert features.grad is not None
    assert torch.all(torch.isfinite(features.grad))
