"""What the engine will actually accept as a prefix match unit."""

import pytest

from kv_aware_router.radix import PrefixTree, validate_match_unit


@pytest.mark.parametrize("unit", [25, 50, 100, 1, 17])
def test_non_multiples_of_the_tensor_core_tile_are_rejected(unit):
    with pytest.raises(ValueError, match="multiple of 16"):
        validate_match_unit(unit)


@pytest.mark.parametrize("unit", [16, 32, 48, 64, 96, 128])
def test_any_multiple_of_16_is_legal_not_just_powers_of_two(unit):
    validate_match_unit(unit)


def test_flashinfer_caps_the_unit_at_64():
    validate_match_unit(64, backend="flashinfer")
    with pytest.raises(ValueError, match="at most 64"):
        validate_match_unit(128, backend="flashinfer")


def test_flash_attn_has_no_ceiling_beyond_the_tile_rule():
    validate_match_unit(256, backend="flash_attn")


def test_unknown_backend_is_an_error_not_a_silent_pass():
    with pytest.raises(ValueError, match="unknown backend"):
        validate_match_unit(16, backend="made_up")


def test_tree_validates_at_construction():
    with pytest.raises(ValueError, match="multiple of 16"):
        PrefixTree(prefix_match_unit=25)
