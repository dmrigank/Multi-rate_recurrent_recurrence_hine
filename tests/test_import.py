"""Smoke test: verify the package and all submodules can be imported."""

import msr_hine
import msr_hine.data.solver
import msr_hine.data.generate
import msr_hine.data.dataset
import msr_hine.spectral.truncation
import msr_hine.models.unet
import msr_hine.models.fno
import msr_hine.models.encoders
import msr_hine.models.recurrence
import msr_hine.models.film
import msr_hine.models.msr_hine
import msr_hine.models.hine
import msr_hine.models.fno_baseline
import msr_hine.losses
import msr_hine.train
import msr_hine.rollout
import msr_hine.metrics
import msr_hine.utils


def test_package_importable():
    assert msr_hine.__version__ == "0.1.0"


def test_all_submodules_importable():
    modules = [
        msr_hine.data.solver,
        msr_hine.data.generate,
        msr_hine.data.dataset,
        msr_hine.spectral.truncation,
        msr_hine.models.unet,
        msr_hine.models.fno,
        msr_hine.models.encoders,
        msr_hine.models.recurrence,
        msr_hine.models.film,
        msr_hine.models.msr_hine,
        msr_hine.models.hine,
        msr_hine.models.fno_baseline,
        msr_hine.losses,
        msr_hine.train,
        msr_hine.rollout,
        msr_hine.metrics,
        msr_hine.utils,
    ]
    for mod in modules:
        assert mod is not None
