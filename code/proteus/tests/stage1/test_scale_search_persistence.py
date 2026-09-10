"""Persistence scale-search tests relocated under ``tests/stage1/persistence/``.

A6-T4 (#28 / #46): the former monolithic ``test_scale_search_persistence.py``
(~45 min) was split into slow-marked modules plus an unmarked smoke slice:

* ``persistence/test_core_smoke.py`` — default unmarked smoke (A6-T5)
* ``persistence/test_within_interval_modes.py`` — slow
* ``persistence/test_multiseed_densify.py`` — slow
* ``persistence/test_phi_half_life.py`` — slow

This file intentionally contains no tests so pytest does not double-collect.
Run the full former suite with::

    pytest tests/stage1/persistence -m slow

Or the default smoke with::

    pytest tests/stage1/persistence/test_core_smoke.py
"""
