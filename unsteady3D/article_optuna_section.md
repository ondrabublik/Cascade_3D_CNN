# Hyperparameter optimization

*(Draft methodology section — no results included yet. `[TODO]` marks numbers to
confirm before submission. Convert to LaTeX later; the generated tables from
`optuna_make_table.py` slot into the places marked below.)*

## Optimization strategy

The hyperparameters of the network were optimized with Optuna [Akiba et al.,
2019], an open-source hyperparameter optimization framework. Candidate
configurations were proposed by the Tree-structured Parzen Estimator (TPE)
sampler, a Bayesian optimization method that models the distributions of
well- and poorly-performing configurations and preferentially samples from
promising regions of the search space. Unpromising trials were terminated
early by a median-stopping rule (pruning): after a warm-up period of 30
epochs, a trial was stopped whenever its intermediate validation error fell
below the median of all previous trials at the same epoch. Early stopping of
this kind typically reduces the total computational cost of a study by a
factor of two to three without materially affecting the final ranking of
configurations.

## Surrogate optimization on a coarser grid

A single full training run on the fine computational grid used in this work
requires on the order of 2,000 epochs to converge, which renders a direct
hyperparameter search — requiring on the order of a hundred training runs —
computationally intractable. The search was therefore carried out on a
coarser variant of the same training data: the fluid domain, the geometry,
and the governing flow conditions are identical to the fine-grid case, and
only the spatial resolution of the interpolation grid differs (48 × 32 × 16
cells, i.e. approximately 64 times fewer cells than the [TODO: confirm]
192 × 128 × 64 fine grid). Because the network is fully convolutional and
operates on normalized fields, its hyperparameters — channel widths, kernel
sizes, activation functions, and the parameters of the optimization procedure
— characterize the learning problem rather than the specific grid resolution,
and the configuration found on the coarse grid is expected to transfer to the
fine grid. A single coarse-grid epoch is roughly two orders of magnitude
cheaper than a fine-grid epoch, which makes a systematic search feasible.

In addition, each trial was trained for a reduced budget of [TODO] 300 epochs
instead of the ~2,000 epochs of a production run. The purpose of the search
is the *relative ranking* of configurations rather than their final converged
accuracy; 300 epochs proved sufficient for the validation error to
discriminate clearly between configurations, and the pruning mechanism
removes hopeless trials long before this limit.

## Network depth

The depth of the U-Net (the number of resolution levels) was deliberately
excluded from the search. Each encoder level halves the grid in every spatial
direction, so a network of depth $d$ requires all three grid dimensions to be
divisible by $2^{d-1}$. The admissible depth is therefore dictated by the
grid, not by the optimizer: on the coarse 48 × 32 × 16 grid the maximum is
$d = 5$ (bottleneck resolution 3 × 2 × 1). Since a deeper network enlarges
the receptive field at negligible additional cost and consistently benefits
the elliptic character of the pressure field, we always employ the maximum
depth the grid admits, and $d = 5$ was used throughout the search, matching
the depth used for the fine-grid production training.

## Two-stage search protocol

Optimizing all hyperparameters simultaneously mixes two groups of parameters
of different character: parameters that define the *architecture* of the
network, and parameters that control the *training procedure*. A joint search
over both groups inflates the dimensionality of the search space, and, more
importantly, obscures the attribution of improvements — a well-chosen
learning rate can mask a poor architecture and vice versa. The search was
therefore split into two consecutive studies:

**Stage A — architecture.** The architectural parameters (Table A) were
searched while all training parameters were held fixed at the reference
values used elsewhere in this work (learning rate $10^{-4}$, gradient
clipping norm 1.0, loss weights $\lambda_v = 1.0$, $\lambda_p = 0.1$).

**Stage B — training procedure.** The architecture was then frozen at the
best configuration found in Stage A, and the parameters of the training
procedure (Table B) were searched.

This sequential (coordinate-descent-like) protocol does not guarantee the
global joint optimum, since architecture and training parameters are not
strictly independent; it does, however, yield a well-conditioned search in
each stage and a clean interpretation of both resulting studies. The
reference training parameters used in Stage A are values already validated in
preliminary experiments, which limits the risk that the architecture ranking
is distorted by an unsuitable training setup.

### Stage A: architectural parameters

| Parameter | Search range | Description |
|---|---|---|
| $n_{\mathrm{ch}}$ (`nChannel`) | 8, 10, …, 32 | Base number of convolutional channels; controls the overall capacity (width) of the network. |
| $w$ (`frameWidth`) | 1, 2, 3 | Half-width of the convolution kernel; the kernel size is $(2w{+}1)^3$, i.e. $3^3$, $5^3$, or $7^3$. Larger kernels enlarge the receptive field per layer at cubically growing cost. |
| $g$ (`growFactor`) | 0, 1 | Channel growth exponent across levels: the number of channels at level $i$ is $n_{\mathrm{ch}} \cdot i^{g}$, i.e. constant width ($g=0$) or linearly growing width ($g=1$) toward the bottleneck. |
| activation (`act`) | ReLU, ELU, swish | Activation function of all hidden convolutional layers. |

The activation of the output layer is not searched: it is fixed to the
logistic sigmoid, because the network predicts normalized fields in $[0, 1]$
that are subsequently denormalized to physical units inside the model.

### Stage B: training parameters

| Parameter | Search range | Description |
|---|---|---|
| $\eta$ (`learningRate`) | $[10^{-5},\ 3\times10^{-3}]$, log-uniform | Learning rate of the Adam optimizer. |
| $c$ (`clipNorm`) | $[0.1,\ 10]$, log-uniform | Global gradient-norm clipping threshold. Clipping stabilizes the multi-step rollout training, in which the gradient propagates through a chain of recurrent applications of the network. |

The loss weights are not searched; they are kept at the reference values
$\lambda_v = 1.0$, $\lambda_p = 0.1$ in both stages.

## Objective function and trial protocol

Each trial trains the multi-step rollout model on the identical,
pre-generated set of training sequences (40 five-step sequences, of which the
last 20 % are held out for validation). The dataset — including the
stochastic perturbation noise injected during its preparation — is generated
once and shared by all trials, so that trials differ exclusively in the
hyperparameters under study.

The training loss is the weighted mean-squared error of the predicted
velocity and pressure fields over the rollout,

$$
\mathcal{L} \;=\; \lambda_v\,\overline{(u - \hat u)^2 + (v - \hat v)^2 + (w - \hat w)^2}
\;+\; \lambda_p\,\overline{(p - \hat p)^2},
$$

where the overbar denotes the mean over all grid points, rollout steps, and
samples, and the weights are fixed at $\lambda_v = 1.0$, $\lambda_p = 0.1$ in
both stages. Since the loss is identical for all trials, its validation value
serves directly as the optimization objective; the objective of a trial is
the best (minimum) validation loss observed over its epochs.

Both studies minimize this objective over [TODO] 100 (Stage A) and [TODO] 100
(Stage B) trials. All remaining settings are common to both stages and
identical to the production training: depth $d = 5$, rollout length of 5
steps, batch size 2, Adam optimizer, and a fixed random seed for the network
initialization, so that trials are not confounded by initialization noise.

## Results

*(To be added: generated tables — search space with best values, top-10
trials, and fANOVA hyperparameter importances for each stage; see
`optuna_results/table_*.tex`.)*

---

### References

- T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama: *Optuna: A
  Next-generation Hyperparameter Optimization Framework.* Proceedings of the
  25th ACM SIGKDD International Conference on Knowledge Discovery & Data
  Mining, 2019.
- J. Bergstra, R. Bardenet, Y. Bengio, B. Kégl: *Algorithms for
  Hyper-Parameter Optimization.* Advances in Neural Information Processing
  Systems 24, 2011. (TPE sampler)
