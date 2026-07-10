# Deploying pulse-search on USTC-SCC

Guide for running the 10-pulse search on the USTC Supercomputing Center
(scc.ustc.edu.cn, 瀚海 Intel Xeon system) and managing the jobs. Commands on
the cluster are `scc$`, on your Mac `local$`. Cluster facts below are the
measured values for this account (from the ustc-scc handbook); re-verify after
a system/account change with `sinfo`, `sacctmgr show assoc user=$USER`, and
`module avail python`.

**Cluster red lines (enforced):** login nodes are for editing, transferring,
and submitting only — all computation goes through `sbatch`/`srun`; no
vscode/jupyter/cursor/proxy processes on any node; the center does **not**
back up data (download results promptly); conda/pip need the Tsinghua mirror
(outbound network is restricted — this also means `git clone` from GitHub
usually fails; use `rsync` from your Mac).

---

## 1. Log in and transfer the code

- SSH only, password + Google Authenticator two-factor; campus IP required
  (off campus: USTC VPN first). ⚠️ 5 wrong passwords in 10 min = your IP is
  banned for 10 min.
- Login nodes: x86 `211.86.151.101` / `211.86.151.102`. File transfer is
  SFTP-based (`scp`/`rsync`/`sftp`).

```bash
local$ ssh <username>@211.86.151.101        # prompts password, then 2FA code

# Copy the repo (excludes local outputs/venv; re-run to push config tweaks)
local$ cd "/Users/kun/Documents/GRAPE VERGE"
local$ rsync -avzP --exclude '.venv' --exclude 'outputs*' \
           --exclude '.git' --exclude '__pycache__' \
           ./ <username>@211.86.151.101:~/grape/
```

Tip: with 2FA every new connection asks for a fresh code. Add SSH connection
reuse to `~/.ssh/config` on your Mac so rsync rides an existing session:

```
Host scc
    HostName 211.86.151.101
    User <username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
```

Then `ssh scc` once (password + code), and subsequent `rsync ... scc:~/grape/`
runs without re-authenticating.

---

## 2. One-time environment setup (login node)

The cluster's `anaconda3/2025.06` module ships numpy/scipy; the project venv
adds the repo's pinned requirements on top. pip must use the Tsinghua mirror:

```bash
scc$ cd ~/grape
scc$ source /etc/profile
scc$ module purge && module load anaconda3/2025.06
scc$ pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
scc$ python3 -m venv .venv
scc$ .venv/bin/pip install --no-cache-dir -r requirements.txt   # --no-cache-dir: home quota ~500GB
```

Sanity-check on a compute node (login nodes must not run computation; the
`test` partition gives 20 minutes, plenty):

```bash
scc$ srun -p test --qos=qos_test -N 1 -c 2 --time=0:15:00 --pty /bin/bash
# now on a compute node:
$ source /etc/profile && module load anaconda3/2025.06 && cd ~/grape
$ .venv/bin/python -m pytest tests/test_quantum_control.py -q     # expect: 70 passed
$ .venv/bin/python -m experiments.spin_boson.pulse_search.pulse_gallery --list   # 10 pulses
$ exit
```

---

## 3. Submit

`submit_slurm.sh` is pre-filled for this system — `CPU-Little` +
`qos_cpu_little` (single-node tasks, 7-day limit, QoS is mandatory and must
match the partition), `-c 10` matching `runtime.workers: 10` in `search.yaml`,
`source /etc/profile` before `module`, BLAS threads pinned to 1. A 40-core
cnode is shared by up to 4 jobs, so `-c 10` array tasks pack efficiently
without `--exclusive`.

```bash
scc$ cd ~/grape
scc$ mkdir -p logs                                        # Slurm won't create the log dir
scc$ sbatch --test-only experiments/spin_boson/pulse_search/submit_slurm.sh   # schedulability check
scc$ sbatch experiments/spin_boson/pulse_search/submit_slurm.sh
Submitted batch job 1234567
```

One array task per gallery pulse (`--array=0-9`); each writes only its own
`experiments/spin_boson/pulse_search/outputs/<pulse_name>/`, and the group folder gets a
`config.yaml` snapshot automatically.

**Another setup** (different noise/gate time): copy + edit the YAML, pick a
fresh group folder, submit with env-var overrides (the script reads both):

```bash
scc$ CONFIG=experiments/spin_boson/pulse_search/search_heating.yaml \
     OUTPUT_DIR=experiments/spin_boson/pulse_search/outputs/heating \
     sbatch experiments/spin_boson/pulse_search/submit_slurm.sh
```

**Long runs**: raise `#SBATCH --time` up to `7-00:00:00` (CPU-Little's cap);
a whole-array cap on concurrency is `--array=0-9%4` (max 4 tasks at once,
gentler on the group's core-hour budget).

---

## 4. Manage running jobs

```bash
scc$ squeue -u $USER                          # PD=pending R=running CG=completing
scc$ speek -f 1234567_0                       # center's command: live output of a running task
scc$ tail -f logs/pulse-search_1234567_0.out  # same thing via the log file
scc$ scontrol show job 1234567_3              # full detail, incl. why a task is pending
```

Pending (`PD`) reasons in `squeue`: `Priority`/`Resources` = normal queueing;
`QOS*Limit` = you hit a QoS cap; `InvalidQOS` = partition/QoS mismatch (fix
and resubmit); `ReqNodeNotAvail` = node maintenance.

**Standings mid-run** — `result.json` appears as each task finishes, so this
works any time (reading JSON on the login node is fine):

```bash
scc$ .venv/bin/python -m experiments.spin_boson.pulse_search.collect_results \
         --output-dir experiments/spin_boson/pulse_search/outputs
```

**Cancel / resubmit:**

```bash
scc$ scancel 1234567                # whole array
scc$ scancel 1234567_3              # one task
scc$ sacct -j 1234567 --format=JobID,State,ExitCode,Elapsed,TotalCPU,MaxRSS,ReqMem
scc$ sbatch --array=3,7 experiments/spin_boson/pulse_search/submit_slurm.sh   # rerun only failures
```

A resubmitted task overwrites only its own `<pulse_name>/` folder. Use the
same mode (optimize vs `--evaluate-only`) as the original run —
`collect_results.py` refuses to summarize a folder with mixed modes.

---

## 5. Bring results home

⚠️ The center does not back up data — pull results as soon as the array
finishes.

```bash
scc$ .venv/bin/python -m experiments.spin_boson.pulse_search.collect_results \
         --output-dir experiments/spin_boson/pulse_search/outputs

local$ cd "/Users/kun/Documents/GRAPE VERGE"
local$ rsync -avzP scc:~/grape/experiments/spin_boson/pulse_search/outputs/ \
           experiments/spin_boson/pulse_search/outputs/
```

The group folder is self-contained: `config.yaml` (exact setup), `summary.md`
(ranking), per-pulse `final_pulse_s<steps>.npz` / `report.md` / `step_log.csv`. Verify
the winner locally against the snapshot config (add `--faithful` for the exact
density-matrix check):

```bash
local$ .venv/bin/python -m experiments.driver.run_experiment evaluate \
           --config experiments/spin_boson/pulse_search/outputs/config.yaml \
           --pulse-npz experiments/spin_boson/pulse_search/outputs/<best_name>/final_pulse_s400.npz
```

---

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `module: command not found` in job log | script must `source /etc/profile` first (already in `submit_slurm.sh` — check custom scripts) |
| `sbatch` rejected with `InvalidQOS` | partition/QoS mismatch; `scontrol show partition CPU-Little` → `AllowQos=`, fix `--qos` |
| job dies instantly, no output | `logs/` didn't exist at submit time (`mkdir -p logs`), or see the `.err` file; `bash -n` the script |
| pip/conda timeouts | Tsinghua mirror not configured (step 2); outbound network is restricted |
| `Disk quota exceeded` | delete old checkpoints/core dumps/`pip` cache; quota ≈ 500 GB, expansion needs an email with justification |
| task `TIMEOUT` in `sacct` | raise `--time` (≤ 7 days on CPU-Little) or lower `optimizer.maxiter` in the YAML |
| `OUT_OF_MEMORY` / `Killed` | check `sacct --format=MaxRSS,ReqMem`; a shared cnode gives ~4.8 GB/core → 10 cores ≈ 48 GB, plenty for `search.yaml`; if exceeded, reduce `n_levels`/`workers` or use the big-memory partition |
| pulse `missing` after array "finished" | gallery grew past `--array` range — submit the missing index alone (`sbatch --array=10 ...`) |
| `--index N out of range` | gallery shrank below the range — re-sync `--array=0-<N-1>` with `pulse_gallery --list` |
| stuck in `CG`, can't cancel | contact the admins (sccadmin@ustc.edu.cn); don't resubmit on top |
| locked out of login | 5 wrong passwords in 10 min bans your IP for 10 min — wait, switch IP, or contact the center |
| password/shell change | web only: http://scc.ustc.edu.cn/user/chpasswd.php (`passwd`/`chsh` are disabled) |

---

## 7. Quick reference

| Action | Command |
| --- | --- |
| Push code/config updates | `rsync -avzP --exclude '.venv' --exclude 'outputs*' ./ scc:~/grape/` |
| Submit the search | `mkdir -p logs && sbatch experiments/spin_boson/pulse_search/submit_slurm.sh` |
| Submit another setup | `CONFIG=<yaml> OUTPUT_DIR=<dir> sbatch experiments/spin_boson/pulse_search/submit_slurm.sh` |
| Queue state | `squeue -u $USER` |
| Live output of task N | `speek -f <jobid>_<N>` or `tail -f logs/pulse-search_<jobid>_<N>.out` |
| Standings so far | `.venv/bin/python -m experiments.spin_boson.pulse_search.collect_results --output-dir <dir>` |
| Cancel array / one task | `scancel <jobid>` / `scancel <jobid>_<N>` |
| Post-mortem | `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS` |
| Rerun failed tasks | `sbatch --array=<i,j> experiments/spin_boson/pulse_search/submit_slurm.sh` |
| Fetch results | `rsync -avzP scc:~/grape/experiments/outputs/<dir>/ experiments/outputs/<dir>/` |
| Interactive debug shell | `srun -p test --qos=qos_test -N 1 -c 2 --time=0:15:00 --pty /bin/bash` |
