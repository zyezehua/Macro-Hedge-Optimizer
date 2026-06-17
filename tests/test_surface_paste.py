import numpy as np
import pytest

from mho.io.surface_paste import parse_iv, parse_moneyness, parse_surface, parse_tenor


def test_parse_tenor():
    assert parse_tenor("1M") == pytest.approx(1 / 12)
    assert parse_tenor("3M") == pytest.approx(0.25)
    assert parse_tenor("2W") == pytest.approx(14 / 365)
    assert parse_tenor("1Y") == pytest.approx(1.0)
    assert parse_tenor("0.5") == pytest.approx(0.5)


def test_parse_moneyness_and_iv():
    assert parse_moneyness("80%") == pytest.approx(0.80)
    assert parse_moneyness("0.8") == pytest.approx(0.80)
    assert parse_moneyness("110") == pytest.approx(1.10)
    assert parse_iv("20%") == pytest.approx(0.20)
    assert parse_iv("0.20") == pytest.approx(0.20)
    assert parse_iv("35") == pytest.approx(0.35)


def test_parse_surface_tab_and_pct():
    text = (
        "M\t1M\t3M\t6M\t1Y\n"
        "80%\t32%\t30%\t28%\t26%\n"
        "100%\t20%\t20%\t20%\t20%\n"
        "110%\t18%\t19%\t20%\t21%\n"
    )
    surf = parse_surface(text)
    assert list(surf.moneyness) == [0.80, 1.00, 1.10]
    assert surf.maturities[0] == pytest.approx(1 / 12)
    assert surf.iv(1.00, 0.5) == pytest.approx(0.20)
    assert surf.iv(0.80, 1 / 12) == pytest.approx(0.32)


def test_parse_surface_csv_and_unsorted():
    text = "label,3M,1M\n100,20,21\n90,24,26\n"
    surf = parse_surface(text)
    # axes should be sorted ascending regardless of paste order
    assert list(surf.moneyness) == [0.90, 1.00]
    assert surf.maturities[0] == pytest.approx(1 / 12)
    assert surf.iv(0.90, 1 / 12) == pytest.approx(0.26)


def test_parse_surface_bad_row():
    with pytest.raises(ValueError):
        parse_surface("M,1M,3M\n80%,20%\n")
