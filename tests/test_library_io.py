"""Round-trip + validation for custom stress-template serialization."""

import pytest

from mho.scenarios.library import HISTORICAL_STRESSES, StressTemplate
from mho.scenarios.library_io import dump_templates, load_templates, template_from_dict
from mho.scenarios.macro import InstrumentShock


def test_roundtrip_preserves_templates():
    out = load_templates(dump_templates(HISTORICAL_STRESSES))
    assert set(out) == set(HISTORICAL_STRESSES)
    for key, tpl in HISTORICAL_STRESSES.items():
        got = out[key]
        assert got.name == tpl.name
        assert got.equity == tpl.equity
        assert got.credit == tpl.credit


def test_roundtrip_custom_template():
    custom = {"my_stress": StressTemplate(
        "my_stress", "My Stress",
        equity=InstrumentShock(-0.30, 0.20, "skew_twist", 0.1),
        credit=InstrumentShock(-0.15, 0.10),
        note="hand-built")}
    out = load_templates(dump_templates(custom))
    assert out["my_stress"].equity.spot_shock == -0.30
    assert out["my_stress"].equity.vol_mode == "skew_twist"
    assert out["my_stress"].credit.spot_shock == -0.15
    assert out["my_stress"].note == "hand-built"


def test_load_rejects_non_list():
    with pytest.raises(ValueError):
        load_templates('{"key": "x"}')


def test_load_rejects_bad_json():
    with pytest.raises(ValueError):
        load_templates("not json{{")


def test_template_from_dict_rejects_missing_field():
    with pytest.raises(ValueError):
        template_from_dict({"key": "x", "name": "y", "equity": {"spot_shock": -0.1}})  # no credit


def test_load_rejects_bad_vol_mode():
    with pytest.raises(ValueError):
        load_templates('[{"key":"x","name":"y","equity":{"spot_shock":-0.1,"vol_mode":"weird"},'
                        '"credit":{"spot_shock":-0.1}}]')
