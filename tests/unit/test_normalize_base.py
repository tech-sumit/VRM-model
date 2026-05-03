from vrm.data.normalize import REGISTRY, NormalizeSpec


def test_all_registered_sources_have_specs():
    expected = {
        "mavis",
        "mathv360k",
        "vision_r1_cold",
        "geo170k",
        "chartqa",
        "mm_eureka",
        "geometry3k",
        "mathvista",
        "we_math",
        "geoqa",
        "tabmwp",
    }
    assert expected.issubset(set(REGISTRY.keys()))


def test_spec_fields_populated():
    for name, spec in REGISTRY.items():
        assert isinstance(spec, NormalizeSpec)
        assert spec.hf_id, f"{name} missing hf_id"
        assert callable(spec.normalize), f"{name} missing normalize fn"
