import aerosandbox as asb
import aerosandbox.numpy as np
from typing import List
import copy
import pytest


def test_alpha_wind():
    op_point = asb.OperatingPoint(
        alpha=90,
        beta=0
    )
    x, y, z = op_point.convert_axes(
        0, 0, 1,
        "geometry",
        "wind"
    )
    assert x == pytest.approx(-1)
    assert y == pytest.approx(0)
    assert z == pytest.approx(0)


def test_beta_wind():
    op_point = asb.OperatingPoint(
        alpha=0,
        beta=90
    )
    x, y, z = op_point.convert_axes(
        0, 1, 0,
        "geometry",
        "wind"
    )
    assert x == pytest.approx(1)
    assert y == pytest.approx(0)
    assert z == pytest.approx(0)


def test_beta_wind_body():
    op_point = asb.OperatingPoint(
        alpha=0,
        beta=90
    )
    x, y, z = op_point.convert_axes(
        0, 1, 0,
        "body",
        "wind"
    )
    assert x == pytest.approx(1)
    assert y == pytest.approx(0)
    assert z == pytest.approx(0)


def test_alpha_stability_body():
    op_point = asb.OperatingPoint(
        alpha=90,
        beta=0
    )
    x, y, z = op_point.convert_axes(
        0, 0, 1,
        "body",
        "stability"
    )
    assert x == pytest.approx(1)
    assert y == pytest.approx(0)
    assert z == pytest.approx(0)


def test_beta_stability_body():
    op_point = asb.OperatingPoint(
        alpha=0,
        beta=90
    )
    x, y, z = op_point.convert_axes(
        0, 1, 0,
        "body",
        "stability"
    )
    assert x == pytest.approx(0)
    assert y == pytest.approx(1)
    assert z == pytest.approx(0)


def test_order_wind_body():
    op_point = asb.OperatingPoint(
        alpha=90,
        beta=90,
    )
    x, y, z = op_point.convert_axes(
        0, 1, 0,
        "body",
        "wind"
    )
    assert x == pytest.approx(1)
    assert y == pytest.approx(0)
    assert z == pytest.approx(0)


if __name__ == '__main__':
    pytest.main()
