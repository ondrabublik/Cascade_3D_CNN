"""Optuna hyperparameter search for the multistep 3D UNet -- two-stage protocol.

Runs on the small (coarse-grid, 48x32x16) dataset in DATA/dataooo. The fluid
domain is identical to the fine-grid article data, only the grid resolution
differs, so the found hyperparameters transfer. Network depth is NOT searched:
it is fixed at the maximum the grid allows (deep=5, every dimension divisible
by 2^4), which is also what the article training uses.

Stage A -- architecture study (--study arch):
    nChannel      int          8 .. 32 (step 2)
    frameWidth    int          1 .. 3   (Conv3D kernel = 2*frameWidth+1)
    growFactor    categorical  0, 1     (constant vs. linearly growing channels)
    act           categorical  relu, elu, swish   (hidden activation)
  Training parameters are fixed to the article defaults (lr=1e-4, clipNorm=1,
  loss weights 1.0/0.1). The output activation stays sigmoid -- the built-in
  denormalization requires outputs in [0, 1].

Stage B -- training study (--study train):
    learningRate        log-uniform  1e-5 .. 3e-3
    clipNorm            log-uniform  0.1 .. 10
  Loss weights are fixed at the article values (velocity 1.0, pressure 0.1).
  The architecture is fixed: taken automatically from the best complete trial
  of the arch study if it exists, otherwise from the article defaults
  (override with --nChannel/--frameWidth/--growFactor/--act).

Objective (both stages): the best value of a FIXED reference metric -- the
weighted u/v/w/p validation MSE with the article weights (velocity 1.0,
pressure 0.1), computed as a Keras metric independently of the trained loss.
With the loss weights fixed everywhere it now equals val_loss, but keeping it
as a separate metric means the objective definition never silently changes if
loss weights are searched again in the future.

Every trial trains on the identical pre-generated multistep .npy files; the
data (including its injected noise) is prepared once and cached on disk.

Usage:
    python optuna_search.py --study arch  --n-trials 100 --epochs 300
    python optuna_search.py --study train --n-trials 100 --epochs 300
    python optuna_search.py --study arch  --n-trials 2 --epochs 3   # smoke test

Both studies live in the SQLite file optuna_unet3d.db next to this script, so
an interrupted run resumes by simply running the same command again. The
database can be copied to another machine and continued there.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # same thread-safety reason as in trainUnet3D_multistep.py

import numpy as np
import optuna
import tensorflow as tf
from keras.optimizers import Adam

from UNetDev3D_one_param import UNetDev as Unet
from dataClass3D_one_param import Data
from trainUnet3D_multistep import build_multistep_rollout_model, weighted_uvwp_mse

# ----------------------------------------------------------------------------
# Fixed (non-searched) setup -- keep identical to the article training.
# ----------------------------------------------------------------------------
DEEP = 5                  # max allowed by the 48x32x16 grid; not searched
N_STEPS = 5
ACT_OUT = "sigmoid"       # denormalization requires outputs in [0, 1]
VELOCITY_LOSS_WEIGHT = 1.0
PRESSURE_LOSS_WEIGHT = 0.1
LEARNING_RATE = 1e-4
CLIP_NORM = 1.0
BATCH_SIZE = 2            # matches how the .npy batch files were generated
VALIDATION_SPLIT = 0.2
SEED = 42

# Article defaults, used by the train study when no arch study result exists.
DEFAULT_ARCH = {"nChannel": 26, "frameWidth": 2, "growFactor": 0, "act": "relu"}

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIRS = [
    str(PROJECT_DIR / "DATA" / "dataooo" / "transformed_0ooo"),
    str(PROJECT_DIR / "DATA" / "dataooo" / "transformed_10ooo"),
    str(PROJECT_DIR / "DATA" / "dataooo" / "transformed_15ooo"),
    str(PROJECT_DIR / "DATA" / "dataooo" / "transformed_20ooo"),
]

STUDY_NAMES = {"arch": "unet3d_arch", "train": "unet3d_train"}
STORAGE = f"sqlite:///{(Path(__file__).resolve().parent / 'optuna_unet3d.db').as_posix()}"

# Fixed reference metric: the article loss (weights 1.0 / 0.1), always computed
# the same way regardless of what loss the trial trains with. Keras derives the
# log keys from the function name: 'ref_loss' and 'val_ref_loss'.
_ref = weighted_uvwp_mse(velocity_weight=VELOCITY_LOSS_WEIGHT,
                         pressure_weight=PRESSURE_LOSS_WEIGHT)


def ref_loss(y_true, y_pred):
    return _ref(y_true, y_pred)


def load_dataset():
    """Prepare (once) and load ALL multistep batch files into RAM.

    The whole small dataset is ~350 MB as float32, so holding it in memory is
    far cheaper than re-reading .npy files every epoch for hundreds of epochs.
    """
    data = Data(DATA_DIRS)

    first_file = data.dataPath / Path("dataIn_multistep_0.npy")
    if not first_file.exists():
        print("Multistep .npy files not found -- preparing them once now.")
        data.prepare_training_data_multistep(nSteps=N_STEPS)

    xs, ys = [], []
    for i in range(data.nBatches):
        xs.append(data.loadDataIn_multistep(i))
        ys.append(data.loadDataOut_multistep(i))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # Same hold-out logic as trainNetMultistep: the LAST files are validation.
    n_val_files = int(round(data.nBatches * VALIDATION_SPLIT))
    n_val_files = min(n_val_files, data.nBatches - 1)
    n_val = n_val_files * data.batchSize

    x_train, x_val = x[:-n_val], x[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]
    print(f"train sequences: {len(x_train)}, validation sequences: {len(x_val)}")

    return x_train, y_train, x_val, y_val, data.scales, (data.nx, data.ny, data.nz, data.dimIn, data.dimOut)


def best_arch_from_study(storage):
    """Architecture for the train study: best arch-study trial, else defaults."""
    try:
        arch_study = optuna.load_study(study_name=STUDY_NAMES["arch"], storage=storage)
        best = arch_study.best_trial  # raises if no complete trial exists
    except Exception:
        print("No finished arch study found -- using article default architecture.")
        return dict(DEFAULT_ARCH), "article defaults"
    params = {k: best.params[k] for k in ("nChannel", "frameWidth", "growFactor", "act")}
    return params, f"best trial #{best.number} of study '{STUDY_NAMES['arch']}'"


class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """Reports the reference metric to Optuna every epoch and prunes."""

    def __init__(self, trial, monitor="val_ref_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        if not np.isfinite(value):
            self.model.stop_training = True
            return
        self.best = min(self.best, value)
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            self.model.stop_training = True
            self.trial.set_user_attr("pruned_at_epoch", epoch + 1)
            raise optuna.TrialPruned(f"pruned at epoch {epoch + 1}")


class ProgressPrint(tf.keras.callbacks.Callback):
    """Compact one-line-per-10-epochs progress so long studies stay readable."""

    def __init__(self, trial_number, epochs, every=10):
        super().__init__()
        self.trial_number = trial_number
        self.epochs = epochs
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every == 0 or epoch == 0:
            logs = logs or {}
            print(
                f"trial {self.trial_number:3d}  epoch {epoch + 1:4d}/{self.epochs}"
                f"  loss={logs.get('loss', float('nan')):.4e}"
                f"  val_ref_loss={logs.get('val_ref_loss', float('nan')):.4e}",
                flush=True,
            )


def make_objective(study_kind, dataset, epochs, fixed_arch=None):
    x_train, y_train, x_val, y_val, scales, (nx, ny, nz, dim_in, dim_out) = dataset

    def objective(trial):
        if study_kind == "arch":
            arch = {
                "nChannel": trial.suggest_int("nChannel", 8, 32, step=2),
                "frameWidth": trial.suggest_int("frameWidth", 1, 3),
                "growFactor": trial.suggest_categorical("growFactor", [0, 1]),
                "act": trial.suggest_categorical("act", ["relu", "elu", "swish"]),
            }
            learning_rate = LEARNING_RATE
            clip_norm = CLIP_NORM
        else:
            arch = fixed_arch
            learning_rate = trial.suggest_float("learningRate", 1e-5, 3e-3, log=True)
            clip_norm = trial.suggest_float("clipNorm", 0.1, 10.0, log=True)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        net = Unet(
            nx, ny, nz, dim_in, dim_out,
            act=arch["act"],
            actOut=ACT_OUT,
            scales=scales,
            frame_width=arch["frameWidth"],
            nChannel=arch["nChannel"],
            deep=DEEP,
            growFactor=arch["growFactor"],
        )
        net.build()
        trial.set_user_attr("n_params", int(net.model.count_params()))

        multistep_model = build_multistep_rollout_model(net.model, n_steps=N_STEPS)
        multistep_model.compile(
            loss=weighted_uvwp_mse(
                velocity_weight=VELOCITY_LOSS_WEIGHT,
                pressure_weight=PRESSURE_LOSS_WEIGHT,
            ),
            optimizer=Adam(learning_rate=learning_rate, clipnorm=clip_norm),
            metrics=[ref_loss],
        )

        pruning_cb = OptunaPruningCallback(trial)
        try:
            multistep_model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                batch_size=BATCH_SIZE,
                shuffle=True,
                epochs=epochs,
                verbose=0,
                callbacks=[pruning_cb, ProgressPrint(trial.number, epochs)],
            )
        finally:
            del multistep_model, net
            tf.keras.backend.clear_session()

        if not np.isfinite(pruning_cb.best):
            # Diverged (NaN loss) before producing a single finite metric value.
            return float("inf")
        return pruning_cb.best

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--study", choices=["arch", "train"], default="arch",
                        help="arch = architecture search, train = training-parameter search")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="TOTAL number of trials in the study (not additional)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--study-name", default=None,
                        help="override the default study name (mainly for smoke tests)")
    parser.add_argument("--storage", default=STORAGE)
    # Optional architecture overrides for --study train:
    parser.add_argument("--nChannel", type=int, default=None)
    parser.add_argument("--frameWidth", type=int, default=None)
    parser.add_argument("--growFactor", type=int, default=None)
    parser.add_argument("--act", default=None)
    args = parser.parse_args()

    print("Python", sys.version.split()[0], "| TF", tf.__version__, "| Optuna", optuna.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPU devices:", gpus if gpus else "none (running on CPU)")

    fixed_arch = None
    if args.study == "train":
        fixed_arch, source = best_arch_from_study(args.storage)
        for key in DEFAULT_ARCH:
            override = getattr(args, key)
            if override is not None:
                fixed_arch[key] = override
                source += f" (+ --{key} override)"
        print(f"train study architecture ({source}): {fixed_arch}")

    dataset = load_dataset()

    study = optuna.create_study(
        study_name=args.study_name or STUDY_NAMES[args.study],
        storage=args.storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,   # never prune the first 10 trials
            n_warmup_steps=30,     # give every trial at least 30 epochs
            interval_steps=5,
        ),
        load_if_exists=True,       # rerunning the same command resumes the study
    )
    study.set_user_attr("fixed", {
        "deep": DEEP, "n_steps": N_STEPS, "actOut": ACT_OUT,
        "velocityLossWeight": VELOCITY_LOSS_WEIGHT,
        "pressureLossWeight": PRESSURE_LOSS_WEIGHT,
        "batch_size": BATCH_SIZE, "epochs_per_trial": args.epochs, "seed": SEED,
        "objective": "best val_ref_loss (weighted val MSE, weights 1.0/0.1)",
        "dataDirs": DATA_DIRS,
        **({"architecture": fixed_arch} if fixed_arch else
           {"learningRate": LEARNING_RATE, "clipNorm": CLIP_NORM}),
    })

    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, args.n_trials - done)
    print(f"Study '{study.study_name}': {done} trials finished, running {remaining} more.")

    study.optimize(make_objective(args.study, dataset, args.epochs, fixed_arch),
                   n_trials=remaining, gc_after_trial=True)

    print("\nBest trial:", study.best_trial.number)
    print("Best val_ref_loss: %.6e" % study.best_value)
    print("Best params:", json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
