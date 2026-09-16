import os
from pathlib import Path

import pytest

from oxford_reference.source_manifest import inspect_case2_archive


def test_supplied_archive_contains_expected_case2_source_contract():
    archive = os.environ.get("OXFORD_UMAT_ZIP")
    if archive is None:
        pytest.skip("OXFORD_UMAT_ZIP is not set to the supplied Oxford archive")

    facts = inspect_case2_archive(Path(archive))

    assert facts.iphase == 2
    assert facts.slip_model == 2
    assert facts.nslip == 12
    assert facts.hardening_model == 3
    assert facts.k1 == 0.129
    assert facts.k2 == 10.0
    assert facts.has_kocks_mecking_expression
