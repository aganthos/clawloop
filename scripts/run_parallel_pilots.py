"""Run multiple ClawLoop pilots concurrently as subprocesses.

Each pilot runs in its own process with its own output dir + log files.
The runner waits for all processes to complete and returns non-zero if any
failed. Real-time streaming to per-pilot log files; summary at end.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _parse_pilot_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--pilot expects NAME=PATH, got: {spec}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="append", required=True,
                        help="NAME=CONFIG_PATH; may be passed multiple times")
    parser.add_argument("--n-iterations", type=int, required=True)
    parser.add_argument("--base-output-dir", type=Path, required=True)
    args = parser.parse_args()

    pilots = [_parse_pilot_spec(p) for p in args.pilot]
    args.base_output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent  # clawloop/
    runner = repo_root / "scripts" / "run_pilot.py"

    procs = []
    for name, cfg_path in pilots:
        out_dir = args.base_output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "pilot.log"
        log_fh = log_path.open("w")
        cmd = [
            sys.executable, "-u", str(runner), str(cfg_path),
            "--n-iterations", str(args.n_iterations),
            "--output-dir", str(out_dir),
        ]
        print(f"[{name}] starting: {' '.join(cmd)} (log: {log_path})")
        p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((name, p, log_fh, log_path))

    # Wait for all.
    start = time.time()
    results = []
    for name, p, log_fh, log_path in procs:
        rc = p.wait()
        log_fh.close()
        elapsed = time.time() - start
        results.append((name, rc, log_path))
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"[{name}] {status} after {elapsed:.1f}s — log: {log_path}")

    any_fail = any(rc != 0 for _, rc, _ in results)
    print("\nSummary:")
    for name, rc, log_path in results:
        print(f"  {name}: rc={rc}  log={log_path}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
