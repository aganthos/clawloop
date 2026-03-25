"""Generate arithmetic training data in SkyRL parquet format."""
from __future__ import annotations

import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/arithmetic")
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--max_val", type=int, default=100)
    args = parser.parse_args()

    import datasets

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def make_examples(n, split):
        rows = []
        for i in range(n):
            a, b = random.randint(1, args.max_val), random.randint(1, args.max_val)
            rows.append({
                "data_source": "synthetic_arithmetic",
                "prompt": [
                    {"role": "system", "content": "Solve arithmetic problems. Put your answer in \\boxed{} format."},
                    {"role": "user", "content": f"What is {a} + {b}?"},
                ],
                "env_class": "arithmetic",
                "reward_spec": {"method": "rule", "ground_truth": str(a + b)},
                "extra_info": {"split": split, "index": i, "a": a, "b": b},
            })
        return datasets.Dataset.from_list(rows)

    train = make_examples(args.train_size, "train")
    test = make_examples(args.test_size, "test")
    train.to_parquet(os.path.join(output_dir, "train.parquet"))
    test.to_parquet(os.path.join(output_dir, "validation.parquet"))
    print(f"Saved {args.train_size} train + {args.test_size} test to {output_dir}")


if __name__ == "__main__":
    main()
