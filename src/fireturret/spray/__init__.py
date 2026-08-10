# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Spray modelling.

**`control/` may import `spray/envelope.py` and nothing else from this package.**
That boundary is enforced by `tests/test_import_boundaries.py`, and it is what
keeps a per-tick ensemble simulation from ever appearing inside the control loop:
control consumes a four-number envelope; whatever produces that envelope — a
legacy constant now, a droplet ensemble later — stays on the far side of it.
"""
