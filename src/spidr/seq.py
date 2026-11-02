"""Integer sequence encoding (A=0, C=1, G=2, T=3). Complement of base b is 3 - b."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

ALPHABET = "ACGT"
_LUT = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate(ALPHABET):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i
_BYTES = np.frombuffer(ALPHABET.encode(), dtype=np.uint8)


def encode(s: str) -> np.ndarray:
    a = _LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
    if (a == 255).any():
        raise ValueError("sequence contains non-ACGT characters")
    return a


def decode(a: np.ndarray) -> str:
    return _BYTES[np.asarray(a, dtype=np.uint8)].tobytes().decode()


def revcomp(a: np.ndarray) -> np.ndarray:
    return (3 - np.asarray(a, dtype=np.uint8))[..., ::-1]


def onehot(a: np.ndarray, dtype=np.float32) -> np.ndarray:
    return np.eye(4, dtype=dtype)[a]


def is_gc(a: np.ndarray) -> np.ndarray:
    return (a == 1) | (a == 2)


def gc_fraction(a: np.ndarray, axis=-1) -> np.ndarray:
    return is_gc(a).mean(axis=axis)


def mutate(a: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Substitute each base with probability `rate` by one of the other three."""
    a = a.copy()
    m = rng.random(a.shape) < rate
    a[m] = (a[m] + rng.integers(1, 4, m.sum())) % 4
    return a


def write_fasta(path: str | Path, seqs: dict[str, np.ndarray], width: int = 80) -> None:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as fh:
        for name, a in seqs.items():
            s = decode(a)
            fh.write(f">{name}\n")
            for i in range(0, len(s), width):
                fh.write(s[i : i + width] + "\n")


def read_fasta(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    seqs, name, chunks = {}, None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = encode("".join(chunks))
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        seqs[name] = encode("".join(chunks))
    return seqs
