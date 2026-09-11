# two_ion_raman experiments

Two ions, each addressed by its own 3-beam Raman set (6 lasers, 2 Raman
transitions, all beams transverse), with all four collective transverse
modes (x-COM, x-rocking, y-COM, y-rocking) in the Hilbert space and 12
control channels (rabi and detuning per beam). Physics and channel conventions are documented in
`physical_systems/two_ion_raman.py`; `configs/example.yaml` is the schema
reference and a fast smoke-test run (n_levels 2, 3 iterations):

```bash
.venv/bin/python -m experiments.driver.run_experiment --config experiments/two_ion_raman/configs/example.yaml
```
