# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Named, independent random streams.

Today's simulator shares ONE `np.random.default_rng(seed)` between every
renderer. That single generator is why the golden fixtures are so brittle:

* `_draw_fire` makes a **branch-dependent** number of draws (its flicker tongues
  are conditional), and `_draw_splash` only runs at all when the control loop
  opened the valve. So the renderer's random stream is a function of the control
  decisions.
* Which means adding one new draw anywhere — or changing when the valve opens —
  re-rolls every subsequent random number in the whole run, and every fixture
  moves at once, with no merge conflict to warn anybody.

`RngBundle` gives each consumer its own stream, derived from the run seed and
the consumer's NAME. Two properties follow, and both are asserted by
`tests/test_rng.py`:

1. **Independence** — draining one stream cannot perturb another.
2. **Order-independence** — streams are keyed by name, not by registration
   order, so declaring a new one does not renumber the existing ones.

Property 2 is why this uses an explicit `spawn_key` derived from a hash of the
name rather than `SeedSequence.spawn(k)`. `spawn(k)` hands out streams by
*index*, so inserting a consumer shifts every later one — reintroducing exactly
the coupling this module exists to remove.

Streams must be **declared** up front. Asking for an undeclared name raises
rather than quietly minting a stream, because a lazily-invented stream at a call
site is how the ordering coupling grows back.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

import numpy as np

__all__ = ["RngBundle", "stream_spawn_key"]


def stream_spawn_key(name: str) -> tuple[int, ...]:
    """A stable, order-independent `spawn_key` for a named stream.

    blake2b rather than the built-in `hash()`, which is randomised per process by
    PYTHONHASHSEED and would make runs irreproducible across invocations. 8 bytes
    is far more than enough to keep our handful of stream names collision-free,
    and `SeedSequence` accepts arbitrarily large integers.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "big"),)


class RngBundle:
    """A fixed set of named, mutually independent generators from one seed.

    >>> bundle = RngBundle(7, ["fire0", "splash"])
    >>> bundle["fire0"] is bundle["fire0"]
    True
    """

    def __init__(self, seed: int, names: Iterable[str]) -> None:
        self.seed = int(seed)
        self._streams: dict[str, np.random.Generator] = {}
        for name in names:
            if name in self._streams:
                raise ValueError(f"duplicate RNG stream name {name!r}")
            self._streams[name] = np.random.default_rng(
                np.random.SeedSequence(entropy=self.seed, spawn_key=stream_spawn_key(name))
            )

    def __getitem__(self, name: str) -> np.random.Generator:
        try:
            return self._streams[name]
        except KeyError:
            raise KeyError(
                f"undeclared RNG stream {name!r}; declared: {sorted(self._streams)}. "
                "Streams must be declared when the bundle is built — inventing one "
                "at a call site is how ordering coupling grows back."
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._streams

    def __iter__(self) -> Iterator[str]:
        return iter(self._streams)

    def __len__(self) -> int:
        return len(self._streams)

    def names(self) -> tuple[str, ...]:
        return tuple(self._streams)
