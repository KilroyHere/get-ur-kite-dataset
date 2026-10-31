"""Which fingers a hand has and how many joints each keeps.

A finger with fewer than 4 joints keeps every link; its joints are locked (made rigid) from the
fingertip inward, so finger length does not change with joint count.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from . import config

_LETTER = {"index": "I", "middle": "M", "ring": "R", "thumb": "T"}
_ID_RE = re.compile(r"^I(\d)-M(\d)-R(\d)-T(\d)$")


@dataclass(frozen=True)
class Topology:
    counts: tuple[int, int, int, int]  # joints kept per finger (index, middle, ring, thumb); 0 = absent

    def __post_init__(self):
        if len(self.counts) != len(config.FINGERS):
            raise ValueError("need one count per finger")
        for c in self.counts:
            if c != 0 and c not in config.JOINT_COUNTS:
                raise ValueError(f"joint count {c} not in {config.JOINT_COUNTS}")
        if len(self.fingers) < config.MIN_FINGERS:
            raise ValueError(f"need at least {config.MIN_FINGERS} fingers")

    @property
    def id(self) -> str:
        return "-".join(f"{_LETTER[f]}{c}" for f, c in zip(config.FINGERS, self.counts))

    @classmethod
    def from_id(cls, s: str) -> "Topology":
        m = _ID_RE.match(s)
        if not m:
            raise ValueError(f"bad topology id {s!r}")
        return cls(tuple(int(g) for g in m.groups()))

    def count(self, finger: str) -> int:
        return self.counts[config.FINGERS.index(finger)]

    @property
    def fingers(self) -> tuple[str, ...]:
        return tuple(f for f, c in zip(config.FINGERS, self.counts) if c > 0)

    @property
    def has_thumb(self) -> bool:
        return self.count("thumb") > 0

    @property
    def n_dof(self) -> int:
        return sum(self.counts)

    def is_locked(self, finger: str, k: int) -> bool:
        """Is the k-th joint (0 = proximal) of a present finger locked?"""
        return k >= self.count(finger)


def enumerate_topologies(include_thumbless: bool = True) -> list[Topology]:
    options = (0,) + config.JOINT_COUNTS
    out = []
    for counts in itertools.product(options, repeat=len(config.FINGERS)):
        if sum(c > 0 for c in counts) < config.MIN_FINGERS:
            continue
        t = Topology(counts)
        if include_thumbless or t.has_thumb:
            out.append(t)
    return out
