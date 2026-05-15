#!/usr/bin/env python3
"""Draw a random sample from edgelist.parquet and save to parquet."""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Sample edgelist.parquet to a smaller parquet file.")
    parser.add_argument("--input", default="temp/edgelist.parquet", help="Input parquet path")
    parser.add_argument("--output", default="temp/rng_sample_edgelist.parquet", help="Output parquet path")
    parser.add_argument("--frac", type=float, default=0.10, help="Sample fraction (0,1]")
    parser.add_argument("--seed", type=int, default=20260126, help="Random seed")
    args = parser.parse_args()

    if not (0.0 < args.frac <= 1.0):
        raise SystemExit(f"--frac must be in (0,1], got {args.frac}")

    df = pd.read_parquet(args.input)
    sampled = df.sample(frac=args.frac, random_state=args.seed)
    sampled.to_parquet(args.output, index=False)
    print(f"Wrote {len(sampled)} rows to {args.output}")


if __name__ == "__main__":
    main()
