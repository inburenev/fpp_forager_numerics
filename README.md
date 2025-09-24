# Repository Overview

This repository contains the code for **numerical simulations of a jump process with a drift**.  
It was used to produce the results presented in the paper

I.N. Burenev, S.N. Majumdar *First-passage properties of the jump process with a drift. Two exactly solvable cases*,  
J. Phys. A: Math. Theor. 58 (2025), 315001  [arXiv:2504.05409](https://arxiv.org/abs/2504.05409)

---

This repository has two main components:

1. **C Program**: The primary tool for simulations.
2. **Jupyter Notebooks**: Used for generating plots.

## FPP: First Passage Properties Simulation

`fpp` is a program used for generating configuration files and
computing the first passage properties of a particle.

---

### Features

- Simulates the trajectory of a particle based on a model defined
  in a configuration file.
- Supports importance sampling to obtain a rate function and
  study the tails of the probability distribution.
- Can be used to study two observables:
    - First passage time (`τ`).
    - Number of jumps before the first passage (`n`).
- Outputs a summary of statistics and a histogram for the
  probability distribution of the observable of interest.

---

### Usage

#### Compilation

To compile the program, use:

```
gcc -O3 fpp.c -o fpp
```

#### Execution

Run the program with:

```
./fpp <configuration_file>
```

For example:

```
./fpp conf.txt
```

---

### Input

The program requires a configuration file that defines the
model, simulation parameters, and output settings. The
configuration file should be structured into three sections:

1. **Model Parameters**: Defines the particle’s behavior.
2. **Simulation Parameters**: Controls the simulation steps
   and sampling.
3. **Result Parameters**: Configures the histogram and output
   settings.

Parameters should be on separate lines.

#### Example Configuration File

```
[model]
X0 = 0
drift = 1.0
tau_distribution = exponential [1]
M_distribution = exponential [1]
trajectory_length = 10000

[simulation]
n_steps = 100
n_changes = -1
importance_sampling = exponential [1]

[result]
observable = n
hist_min = 0
hist_max = 10000
n_bins = 1001
```

---

### Configuration File Details

#### Model Parameters (`[model]`)

- `X0`: The initial coordinate of the particle.
- `drift`: Drift velocity.
- `tau_distribution`: Distribution for the waiting times.
- `M_distribution`: Distribution for the jumps.
- `trajectory_length`: Maximum length of the trajectory.

**Supported Distributions**:
- `exponential [lambda]`: The density is $begin:math:text$ \\lambda e^{-\\lambda t} $end:math:text$.
- `half_gaussian [lambda]`: Half Gaussian distribution.
- `uniform [lambda]`: Uniform distribution.
- `fixed [lambda]`: Deterministic case.

---

#### Simulation Parameters (`[simulation]`)

- `n_steps`: Number of trajectories to generate.
- `n_changes`: Number of jumps in the trajectory that are
  changed in the Metropolis step. If `n_changes = -1`, all
  trajectories are generated from scratch.
- `importance_sampling`: The importance sampling scheme.
  Supported values:
    - `importance_sampling = none`: Unbiased distribution.
    - `importance_sampling = exponential [theta]`: Exponential
      tilt in the observable. Theta is quasi-temperature.

---

#### Result Parameters (`[result]`)

- `observable`: The observable to monitor:
    - `observable = n`: Number of jumps.
    - `observable = T`: Lifetime of the particle.
- `hist_min`, `hist_max`, `n_bins`: Define the minimum and
  maximum values of the histogram and the number of bins.

---

## Output File

The output file contains aggregated statistics for the
simulation, followed by a detailed histogram.

#### Example Output File

```
# X0: 126.170000
# n_steps: 1000000
# acc: 1000000
# overshoot: 0
# theta_is: 0.000000
# observable: n
# mean_n: 1830.272094
# variance_n: 7850067.172611
# mean_tau: 926.944019
# variance_tau: 1998720.530548
# T_trust: 4910.450587
0.000000  0  0.000000  13.815511
1.000000  0  0.000000  13.815511
...
```

---

### Explanation

1. **Summary Statistics**: Lines starting with `#` contain
   metadata and aggregated statistics:
    - `mean_n` and `variance_n`: Mean and variance of `n`.
    - `mean_tau` and `variance_tau`: Mean and variance of `τ`.
    - `acc`: Number of Metropolis steps accepted.
    - `overshoot`: Number of Metropolis steps where no first
      passage occurred within `trajectory_length`.
    - `T_trust`: Minimal time among trajectories that have not
      crossed the origin after the maximum number of jumps.

2. **Histogram Data**: Detailed histogram for the probability
   distribution of the observable (`n` or `τ`):
    - **Column 1**: Bin centers.
    - **Column 2**: Number of events in each bin.
    - **Column 3 and 4**: Probability components. The true
      probability is computed as:
      ```
      probability = column3 * exp(-column4)
      ```

**Note**: If `overshoot > 0`, the probabilities do not sum to one.
