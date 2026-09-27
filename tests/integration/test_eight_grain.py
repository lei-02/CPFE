import json

import numpy as np

from pfem_cpfe.io.results import load_result, save_result


def test_result_contains_required_full_fields(tmp_path, solved_eight_grain):
    path = tmp_path / "result.npz"

    npz_path, json_path = save_result(path, solved_eight_grain)
    data = load_result(path)

    assert npz_path == path
    assert json_path.exists()
    assert {
        "u",
        "stress",
        "strain",
        "slip",
        "orientation",
        "fp",
        "ssd",
        "crss",
        "residual",
        "timing",
    } <= set(data)
    assert data["stress"].shape == (64, 8, 6)
    assert data["ssd"].shape == (64, 8, 12)
    assert np.all(np.isfinite(data["orientation"]))


def test_result_metadata_records_schema_and_increment_history(
    tmp_path, solved_eight_grain
):
    path = tmp_path / "result"

    _, json_path = save_result(path, solved_eight_grain)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))

    assert metadata["schema_version"] == 1
    assert metadata["converged"] is True
    assert metadata["iterations"] == [0]
    assert set(metadata["timing"]) == {"newton", "end_to_end"}
