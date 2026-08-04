# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Adaptation: estimating what the model got wrong, carefully.

Everything in this package writes to FEEDFORWARD terms only — the opening
solution, a pre-offset, an air-density correction. Nothing here is allowed write
access to the closed-loop correction law in `control/suppress.py`, which is
enforced by `tests/test_import_boundaries.py`.

The reason is the failure mode: an estimator that can move the loop's fixed point
can converge confidently to the wrong place, and a deleted or poisoned estimator
should degrade to exactly today's behaviour rather than to something new.
"""
