"""Shared fixtures.

The battery's values live in :mod:`bess.config`, so tests read them from there
rather than restating them. A changed assumption is then exercised by the suite
instead of quietly diverging from it.

Variations use :func:`dataclasses.replace`, which re-runs ``__post_init__`` and
so keeps validation in force.
"""

from __future__ import annotations

import pytest

from bess.asset.battery import BatterySpec
from bess.config import CONFIG, Config


@pytest.fixture(scope="session")
def config() -> Config:
    """The project's real configuration."""
    return CONFIG


@pytest.fixture
def spec() -> BatterySpec:
    """The configured battery: 200 MWh / 50 MW, 81% round-trip, starting empty."""
    return CONFIG.battery
