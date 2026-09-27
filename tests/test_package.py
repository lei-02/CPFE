def test_package_exposes_version():
    import pfem_cpfe

    assert pfem_cpfe.__version__ == "0.1.0"
