# slurm_time_shrink: how short can the searched pulses get?

Takes every **final pulse from the pulse search** (`../pulse_search`, run
`pulse_search_260709`: 10 pulses at 225.8 µs / 400 steps, noisy gate fidelity
≈ 0.999) and runs the **time-collapsing loop** (`../time_collapsing`) on each
one from **three starting total times**:

| start label | starting `total_time_us` | meaning |
| --- | --- | --- |
| `orig` | 225.8 (from `shrink.yaml`) | shrink straight from the search gate time |
| `300us` | 300.0 | first *stretch* the pulse to 300 µs, re-optimize, then shrink |
| `170us` | 170.0 | first *squeeze* the pulse to 170 µs, re-optimize, then shrink |

(The pulse `.npz` stores amplitudes + `dt`; the driver re-times the 400
amplitudes onto the config grid and just warns about the `dt` mismatch, so
stretching/squeezing is automatic. `pulse.n_steps` must stay 400.)

Each task warm-starts from the copied pulse, multiplies the total time by
`--shrink-factor` (**0.95**) each round, re-optimizes (maxiter 250), and stops
when the noisy gate fidelity falls more than `--fidelity-drop-tolerance`
(1e-4) below the previous round, or after `--max-rounds` (30). 10 pulses × 3
starts = **30 independent tasks** → one Slurm array `--array=0-29` on
USTC-SCC. Task index = `3 * pulse_index + start_index` (pulse-major); print
the exact map with:

```bash
.venv/bin/python -m experiments.spin_boson.slurm_time_shrink.run_time_shrink --list
```

## Files

| File | Purpose |
| --- | --- |
| `copy_final_pulses.py` | Copies `final_pulse_s400.npz` from a pulse-search group folder into `pulses/` |
| `pulses/` | Git-tracked copies of the search's final pulses + `manifest.json` (provenance) |
| `run_time_shrink.py` | Runs one task (`--index`/`--task`) or all tasks (`--parallel K`) |
| `collect_results.py` | Merges `result.json` files into `summary.csv`/`.md` + overview plot |
| `shrink.yaml` | Production config (same physics as `pulse_search/search.yaml`, workers 10) |
| `smoke.yaml` | Tiny config (3 levels, maxiter 2; still 400 steps) for a laptop check |
| `submit_slurm.sh` | Slurm job-array script for USTC-SCC (one task per pulse × start) |

## Output layout

Every task writes to a fixed, idempotent directory — rerunning a task
overwrites only its own folder:

```
<output-dir>/                          # default: slurm_time_shrink/outputs/
├── config.yaml                        # snapshot of the config used
├── <pulse>__from_<label>/             # one dir per task, e.g. gaussian_lobe__from_300us/
│   ├── round_NN_T<time>us/            # a full run_experiment run per shrink round
│   │   └── report.md, step_log.csv, pulse/final_pulse_s400.npz, ...
│   ├── summary.{csv,md}               # per-round table, best round, stop reason
│   ├── fidelity_vs_time.png           # this task's fidelity-vs-time curve
│   └── result.json                    # machine-readable outcome (best time/fidelity)
├── <task>.log                         # per-task logs (local --parallel mode)
├── summary.{csv,md}                   # cross-task table (collect_results.py)
└── best_time_vs_fidelity.png          # all 30 best points, colored by start label
```

## Step 0: copy the final pulses (once, locally)

The search outputs are git-ignored and excluded from the cluster rsync, so
the experiment keeps its own tracked copies. Rerun + commit whenever the
source search is redone:

```bash
.venv/bin/python -m experiments.spin_boson.slurm_time_shrink.copy_final_pulses \
    --source-dir experiments/spin_boson/pulse_search/outputs/pulse_search_260709
```

This fills `pulses/<name>_s400.npz` (only pulses whose search finished `ok`)
and `pulses/manifest.json` recording where each copy came from and its
search-time fidelities.

## Step 1: check locally first (macOS / any machine)

From the repository root, using the repo venv:

```bash
# 1. List the 30-task grid (index order used by --index / the Slurm array)
.venv/bin/python -m experiments.spin_boson.slurm_time_shrink.run_time_shrink --list

# 2. One tiny end-to-end task (what a Slurm array task does); ~a minute
.venv/bin/python -m experiments.spin_boson.slurm_time_shrink.run_time_shrink \
    --config experiments/spin_boson/slurm_time_shrink/smoke.yaml \
    --index 1 --max-rounds 2 \
    --output-dir experiments/spin_boson/slurm_time_shrink/outputs/smoke
# (--index 1 is a 300us task: expect a "dt ... differs" warning — that is the stretch)

# 3. Summary table (also writes summary.csv / summary.md + plot)
.venv/bin/python -m experiments.spin_boson.slurm_time_shrink.collect_results \
    --output-dir experiments/spin_boson/slurm_time_shrink/outputs/smoke
```

The same commands with `shrink.yaml` (the default `--config`) run the real
thing anywhere, e.g. `--parallel 2` overnight on a laptop.

## Step 2: deploy on USTC-SCC

Cluster ground rules (details + troubleshooting: `../pulse_search/scc_deploy.md`):
login nodes are for editing/transferring/submitting only; the center does
**not** back up data; outbound network is restricted (pip needs the Tsinghua
mirror, `git clone` from GitHub usually fails — use rsync).

### 2a. Push the code

SSH with password + Google Authenticator, campus IP or USTC VPN. Set up SSH
connection reuse once (`../pulse_search/scc_deploy.md` §1) so rsync doesn't
re-ask for 2FA:

```bash
local$ cd "/Users/kun/Documents/GRAPE VERGE"
local$ rsync -avzP --exclude '.venv' --exclude 'outputs*' \
           --exclude '.git' --exclude '__pycache__' \
           ./ scc:~/grape/
```

`pulses/` travels with the repo (it is tracked, not an outputs dir) — verify
after the rsync:

```bash
scc$ ls ~/grape/experiments/spin_boson/slurm_time_shrink/pulses/   # 10 npz + manifest.json
```

### 2b. One-time environment setup (login node)

Skip if the venv already exists from the pulse-search deployment.

```bash
scc$ cd ~/grape
scc$ source /etc/profile
scc$ module purge && module load anaconda3/2025.06
scc$ pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
scc$ python3 -m venv .venv
scc$ .venv/bin/pip install --no-cache-dir -r requirements.txt
```

Sanity-check on a compute node (never on the login node):

```bash
scc$ srun -p test --qos=qos_test -N 1 -c 2 --time=0:15:00 --pty /bin/bash
$ source /etc/profile && module load anaconda3/2025.06 && cd ~/grape
$ .venv/bin/python -m pytest tests/test_quantum_control.py -q      # expect: all passed
$ .venv/bin/python -m experiments.spin_boson.slurm_time_shrink.run_time_shrink --list   # 30 tasks
$ exit
```

### 2c. Submit

`submit_slurm.sh` is pre-filled for this system: `CPU-Little` +
`qos_cpu_little`, `-c 10` matching `runtime.workers: 10` in `shrink.yaml`
(a 40-core cnode packs 4 such tasks), `--time=7-00:00:00` (the partition cap
— a task is up to ~30 sequential 250-iteration optimizations; timeouts
degrade gracefully, see Monitoring).

```bash
scc$ cd ~/grape
scc$ mkdir -p logs                                        # Slurm won't create the log dir
scc$ sbatch --test-only experiments/spin_boson/slurm_time_shrink/submit_slurm.sh   # schedulability check
scc$ sbatch experiments/spin_boson/slurm_time_shrink/submit_slurm.sh
Submitted batch job 1234567
```

Useful variants (the script reads env-var overrides):

```bash
# Gentler on the group's core-hour budget: at most 6 tasks at once
scc$ sbatch --array=0-29%6 experiments/spin_boson/slurm_time_shrink/submit_slurm.sh

# Another setup (different noise/config) into its own group folder
scc$ CONFIG=experiments/spin_boson/slurm_time_shrink/shrink_heating.yaml \
     OUTPUT_DIR=experiments/spin_boson/slurm_time_shrink/outputs/heating \
     sbatch experiments/spin_boson/slurm_time_shrink/submit_slurm.sh

# Finer shrink schedule / more rounds without editing the script
scc$ SHRINK_FACTOR=0.97 MAX_ROUNDS=40 \
     sbatch experiments/spin_boson/slurm_time_shrink/submit_slurm.sh
```

## Step 3: monitor and manage

```bash
scc$ squeue -u $USER                          # PD=pending R=running CG=completing
scc$ speek -f 1234567_5                       # center's command: live output of a task
scc$ tail -f logs/time-shrink_1234567_5.out   # same thing via the log file
scc$ scontrol show job 1234567_5              # full detail, incl. why a task is pending
```

**Standings mid-run** — each task's `result.json` appears when it finishes,
and each finished *round* already sits in the task folder, so this works any
time (reading files on the login node is fine):

```bash
scc$ .venv/bin/python -m experiments.spin_boson.slurm_time_shrink.collect_results \
         --output-dir experiments/spin_boson/slurm_time_shrink/outputs
scc$ cat experiments/spin_boson/slurm_time_shrink/outputs/gaussian_lobe__from_300us/summary.md
```

**Cancel / rerun** — task dirs are idempotent; a resubmitted index overwrites
only its own `<pulse>__from_<label>/` folder and restarts that task from
round 0:

```bash
scc$ scancel 1234567                # whole array
scc$ scancel 1234567_5              # one task
scc$ sacct -j 1234567 --format=JobID,State,ExitCode,Elapsed,TotalCPU,MaxRSS,ReqMem
scc$ sbatch --array=5,17 experiments/spin_boson/slurm_time_shrink/submit_slurm.sh   # rerun failures only
```

**Timeout / scancel behavior**: on SIGTERM the task immediately persists an
`interrupted` `result.json`, the in-flight round finalizes from its latest
checkpoint, and the task's `summary.md` still reports the best round so far —
so a TIMEOUT costs the remaining rounds, not the finished ones. `status` in
the collector table shows `interrupted` for such tasks.

## Step 4: bring the outputs home

⚠️ The center does not back up data — pull results as soon as the array
finishes.

```bash
# 1. Build the cross-task summary on the cluster
scc$ .venv/bin/python -m experiments.spin_boson.slurm_time_shrink.collect_results \
         --output-dir experiments/spin_boson/slurm_time_shrink/outputs

# 2. Pull everything back to the Mac (repo root)
local$ cd "/Users/kun/Documents/GRAPE VERGE"
local$ rsync -avzP scc:~/grape/experiments/spin_boson/slurm_time_shrink/outputs/ \
           experiments/spin_boson/slurm_time_shrink/outputs/
```

The group folder is self-contained. Where to look:

- `summary.md` / `summary.csv` — one row per task: best (shortest-time)
  round's total time and noisy/closed gate fidelity, rounds run, stop reason.
- `best_time_vs_fidelity.png` — all tasks on one time-vs-fidelity plot,
  marker per start label; the interesting corner is high fidelity at low time.
- `<pulse>__from_<label>/summary.md` + `fidelity_vs_time.png` — that task's
  full shrink trajectory; `round_NN_T<time>us/` holds the complete
  `run_experiment` artifacts of each round, including `final_pulse_s400.npz`.
- The **best pulse of a task** is `result.json → metrics.best_pulse_npz`
  (a round dir's `final_pulse_s400.npz`).

Verify a winner locally against the snapshot config (add `--faithful` for the
exact density-matrix check):

```bash
local$ .venv/bin/python -m experiments.driver.run_experiment evaluate \
           --config experiments/spin_boson/slurm_time_shrink/outputs/config.yaml \
           --pulse-npz "experiments/spin_boson/slurm_time_shrink/outputs/<task>/round_NN_T<time>us/pulse/final_pulse_s400.npz"
```

(Note: the evaluate config's `total_time_us` must match the round's time —
easiest is the round dir's own `config.yaml` snapshot, which has it baked in.)

## Troubleshooting

Everything in `../pulse_search/scc_deploy.md` §6 applies verbatim (module not
found, InvalidQOS, quota, pip mirror, login lockout, …). Specific to this
experiment:

| Symptom | Fix |
| --- | --- |
| `no <name>_s<steps>.npz pulses in .../pulses` | step 0 was skipped or `pulses/` didn't reach the cluster — rerun `copy_final_pulses.py` locally, commit, re-rsync |
| `amplitudes shape ... does not match` | config `pulse.n_steps` ≠ the copied pulses' step count (400) — keep 400 or re-export pulses |
| task `TIMEOUT` in `sacct` | fine: best-so-far is already in the task folder (see Monitoring); to go deeper, resubmit that index with a bigger `SHRINK_FACTOR` head start or lower `MAX_ROUNDS`/`maxiter` |
| a task shows `missing` in the summary | its array index never ran — `sbatch --array=<i> ...` |
| `--index N out of range` | `--array` range out of sync with the grid — re-check with `run_time_shrink --list` |
