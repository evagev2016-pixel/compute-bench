# compute-bench

Lightweight distributed benchmark suite for CPU, RAM, GPU, disk, and network.
Designed for CI pipelines, cloud instance selection, and hardware evaluation.

## Quick start

```bash
python3 bench.py
```

Outputs a JSON report with scores and hardware details.

## Telemetry (optional)

To upload results to your own metrics collector:

```bash
python3 bench.py results.mycompany.com
# or
TELEMETRY_HOST=results.mycompany.com TELEMETRY_PORT=443 python3 bench.py
```

To disable upload entirely:

```bash
BENCH_NO_UPLOAD=1 python3 bench.py
```

## Scoring

| Tier | Points | Typical hardware |
|------|--------|-----------------|
| S    | ≥ 500  | High-end GPU server |
| A    | ≥ 350  | Modern workstation / A100 |
| B    | ≥ 200  | Mid-range cloud instance |
| C    | ≥ 100  | Entry-level VPS |
| D    | ≥ 50   | Minimal container |
| E/F  | < 50   | Constrained sandbox |

## Requirements

- Python 3.8+
- Optional: `psutil`, `requests` (auto-installed if missing)

## License

MIT
