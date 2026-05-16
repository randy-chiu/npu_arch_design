"""Convert a little-endian firmware binary into one-word-per-line readmemh."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--words", type=int, default=8192)
    args = parser.parse_args()

    data = args.input.read_bytes()
    if len(data) > args.words * 4:
        raise ValueError(f"{args.input} has {len(data)} bytes, exceeds {args.words * 4} byte ROM")
    padded = data + bytes((-len(data)) % 4)
    words = [
        int.from_bytes(padded[idx : idx + 4], byteorder="little")
        for idx in range(0, len(padded), 4)
    ]
    words.extend([0] * (args.words - len(words)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{word:08x}\n")


if __name__ == "__main__":
    main()
