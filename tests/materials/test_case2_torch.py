import numpy as np
import torch

from pfem_cpfe.materials.case2_torch import case2_step_torch


def test_torch_backend_matches_numpy_backend(case2_batch):
    torch_out = case2_step_torch(**case2_batch.torch_inputs)
    numpy_out = case2_batch.numpy_results

    assert torch.all(torch_out.converged)
    assert torch.all(torch_out.reason_codes == 0)
    np.testing.assert_allclose(
        torch_out.stress.detach().cpu(), numpy_out.stress, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        torch_out.state.ssd.detach().cpu(), numpy_out.state.ssd, rtol=2e-6, atol=2e-8
    )
    np.testing.assert_allclose(
        torch_out.state.fp.detach().cpu(), numpy_out.state.fp, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        torch_out.state.orientation.detach().cpu(),
        numpy_out.state.orientation,
        rtol=2e-6,
        atol=2e-7,
    )


def test_stress_is_differentiable_with_respect_to_deformation(case2_batch):
    deformation = case2_batch.F_np1_double.requires_grad_(True)

    assert torch.autograd.gradcheck(
        case2_batch.stress_function,
        (deformation,),
        eps=1e-6,
        atol=2e-4,
        rtol=2e-3,
    )


def test_invalid_point_is_reported_without_dropping_batch(case2_batch):
    inputs = dict(case2_batch.torch_inputs)
    deformation = inputs["F_np1"].clone()
    deformation[1] = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], dtype=deformation.dtype, device=deformation.device)
    )
    inputs["F_np1"] = deformation

    result = case2_step_torch(**inputs)

    assert result.stress.shape == (2, 6)
    assert result.converged.tolist() == [True, False]
    assert result.reason_codes[1] != 0
    torch.testing.assert_close(result.state.ssd[1], inputs["state_n"].ssd[1])
