import json

from pfem_cpfe.training.train import PatchTrainingConfig, train_patch


def test_one_sample_patch_training_reduces_residual_without_labels(tmp_path):
    config = PatchTrainingConfig(
        output=tmp_path / "patch",
        steps=150,
        learning_rate=2.0e-4,
        seed=20260915,
        hidden=16,
        layers=1,
        heads=4,
        slices=4,
    )

    metrics = train_patch(config)

    assert metrics["final_relative_residual"] < (
        metrics["initial_relative_residual"] / 100.0
    )
    assert (config.output / "model.pt").exists()
    saved = json.loads((config.output / "metrics.json").read_text(encoding="utf-8"))
    assert saved["uses_solution_labels"] is False
