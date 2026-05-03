from vrm.data.normalize import REGISTRY, NormalizeSpec


def test_all_registered_sources_have_specs():
    # Audit 2026-05-03 removed mm_eureka, mathvista, we_math, tabmwp from
    # the registry -- see src/vrm/data/normalize/__init__.py for rationale.
    expected = {
        "mavis",
        "mathv360k",
        "vision_r1_cold",
        "geo170k",
        "chartqa",
        "geometry3k",
        "geoqa",
    }
    assert expected == set(REGISTRY.keys())


def test_spec_fields_populated():
    for name, spec in REGISTRY.items():
        assert isinstance(spec, NormalizeSpec)
        assert spec.hf_id, f"{name} missing hf_id"
        assert callable(spec.normalize), f"{name} missing normalize fn"
