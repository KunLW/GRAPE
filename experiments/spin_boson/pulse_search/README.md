# Pulse search: parallel multi-initial-pulse optimization

GRAPE converges to a local optimum near its initial guess, so the final gate
fidelity depends strongly on the initial pulse. This experiment searches over a
**gallery of named initial pulses**, running one independent optimization per
pulse (identical YAML config, different starting point), then collects all
results into a best-first summary so the best basin can be picked.

Every run is self-contained and writes to a fixed directory
`<output-dir>/<pulse_name>/` (standard run artifacts + `result.json`), which
makes the runs idempotent and safe to launch concurrently — locally via
subprocesses or on a cluster via a Slurm job array.

## Files

| File | Purpose |
| --- | --- |
| `pulse_gallery.py` | The 10 named initial pulses (registry; index order = Slurm array id) |
| `preview_pulses.py` | One-figure preview of all gallery pulses with fidelities + L1/L2 |
| `run_search.py` | Runs one pulse (`--index`/`--pulse`) or all pulses (`--parallel K`) |
| `collect_results.py` | Merges `result.json` files into a sorted table + `summary.csv`/`.md` |
| `search.yaml` | Production config (400 steps, maxiter 200, fluctuations on) |
| `smoke.yaml` | Tiny config for checking the pipeline on a laptop |
| `submit_slurm.sh` | Slurm job-array script for USTC-SCC (one task per pulse) |
| `scc_deploy.md` | Full USTC-SCC deployment & task-management guide |
| `pulses/n<steps>/` | Pre-exported gallery `.npz` files per grid size, for reuse |

## One folder per setup

Everything belonging to one setup — one config + one group of initial pulses —
lives in a single group folder, chosen with `--output-dir` (same flag on
`preview_pulses.py`, `run_search.py`, and `collect_results.py`):

```
<output-dir>/                     # one folder per setup
├── config.yaml                   # snapshot of the config used (written automatically)
├── preview/
│   └── preview.{png,md,csv}      # initial-pulse preview for this config
├── <pulse_name>/                 # one run dir per gallery pulse
│   ├── result.json, report.md, final_pulse_s<steps>.npz, step_log.csv, ...
├── <pulse_name>.log              # per-pulse logs (local --parallel mode)
└── summary.{csv,md}              # written by collect_results.py
```

Start a new noise scenario / gate time / pulse group by picking a fresh
`--output-dir` (e.g. `experiments/spin_boson/pulse_search/outputs/heating`) and passing
the same `--config` to every command. Keep evaluate-only sweeps in their own
folder — `collect_results.py` refuses to mix evaluate and optimize results.

## Check locally first (macOS / any machine)

From the repository root, using the repo venv:

```bash
# 1. List the gallery (index order used by --index / the Slurm array)
.venv/bin/python -m experiments.spin_boson.pulse_search.pulse_gallery --list

# 2. Fast pipeline check: evaluate all pulses (no optimization), 2 at a time
.venv/bin/python -m experiments.spin_boson.pulse_search.run_search \
    --config experiments/spin_boson/pulse_search/smoke.yaml --evaluate-only --parallel 2 \
    --output-dir experiments/spin_boson/pulse_search/outputs/smoke_eval

# 3. One tiny end-to-end optimization (what a Slurm array task does)
.venv/bin/python -m experiments.spin_boson.pulse_search.run_search \
    --config experiments/spin_boson/pulse_search/smoke.yaml --index 0 \
    --output-dir experiments/spin_boson/pulse_search/outputs/smoke

# 4. Summary table (also writes summary.csv / summary.md)
.venv/bin/python -m experiments.spin_boson.pulse_search.collect_results \
    --output-dir experiments/spin_boson/pulse_search/outputs/smoke_eval
```

The same commands with `search.yaml` (the default `--config`) run the
production search on any machine, e.g. `--parallel 2` overnight on a laptop.

### Preview the gallery

One figure + one table with every initial pulse's shape, close/open gate
fidelity, and L1/L2 smoothness penalties (the weighted values the optimizer
subtracts), sorted best-first — no optimization, no per-pulse run dirs.

```bash
# Real numbers (search.yaml is the default config; a few minutes)
.venv/bin/python -m experiments.spin_boson.pulse_search.preview_pulses

# Fast pipeline check (toy physics, seconds)
.venv/bin/python -m experiments.spin_boson.pulse_search.preview_pulses \
    --config experiments/spin_boson/pulse_search/smoke.yaml \
    --output-dir experiments/spin_boson/pulse_search/outputs/smoke
```

Outputs land in `<output-dir>/preview/` inside the group folder
(`preview.png`, `preview.md`, `preview.csv`), next to the `config.yaml`
snapshot. Options: `--workers K` overrides `runtime.workers` for the fidelity
evaluation (handy to keep a laptop responsive); `--config` previews the
gallery under any setup — fidelities and penalty weights follow the config.

Reading the table: `close_gate` is the noise-free fidelity, `open_gate` the
optimizer's noisy objective, `l1`/`l2` the weighted smoothness penalties — a
good starting pulse has high `open_gate` and small penalties.

## Deploy on USTC-SCC (Slurm)

See **[`scc_deploy.md`](scc_deploy.md)** — the complete guide to deploying,
submitting, monitoring, and managing pulse-search jobs on USTC-SCC, and
bringing the results home.

## Reusable pulse .npz exports

`pulses/` holds every gallery pulse pre-exported for several grid sizes
(`n40`, `n100`, `n200`, `n400`, `n800`; files are named
`<pulse>_s<steps>.npz` per the repo convention and store `amplitudes` in
rad/s plus `dt` for the default 225.8 µs gate). Use them anywhere the driver
accepts a pulse file — pick the subfolder matching the config's
`pulse.n_steps`:

```bash
# as the initial pulse of a single optimization run
.venv/bin/python -m experiments.driver.run_experiment \
    --config experiments/spin_boson/pulse_search/search.yaml \
    --initial-pulse-npz experiments/spin_boson/pulse_search/pulses/n400/flattop_s400.npz

# or just evaluate one
.venv/bin/python -m experiments.driver.run_experiment evaluate \
    --config experiments/spin_boson/pulse_search/smoke.yaml \
    --pulse-npz experiments/spin_boson/pulse_search/pulses/n40/flattop_s40.npz
```

They are deterministic, so regenerate (e.g. after adding a pulse or for a new
grid size) with:

```bash
.venv/bin/python -m experiments.spin_boson.pulse_search.pulse_gallery \
    --write-dir experiments/spin_boson/pulse_search/pulses --n-steps 40 100 200 400 800
```

(`run_search.py` does not need these — it builds pulses in-memory at the
config's grid.)

## Adding pulse #11

1. Add one decorated function in `pulse_gallery.py` returning
   `(alpha1_khz, alpha2_khz)` profiles over the normalized time axis `t`:

   ```python
   @pulse("my_shape")
   def _my_shape(t):
       return 30.0, 120.0 * np.sin(np.pi * t) ** 2
   ```

   Bounds clipping, alpha2 endpoint zeroing, kHz→rad/s conversion, and
   validation are applied automatically.

2. Bump the array range in `submit_slurm.sh` (`--array=0-10` for 11 pulses).

New entries are appended, so existing indices — and finished runs — keep their
meaning.
