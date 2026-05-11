"""
Optuna hyperparameter optimization for UNet3D model.

Usage:
    python optuna_optimize.py

Results are stored in an SQLite database (optuna_unet3d.db) so optimization
can be resumed if interrupted.
"""

import sys
import os
import math
import time
import contextlib
import io
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam, AdamW, SGD
from pathlib import Path
import json
import optuna
from optuna.integration import TFKerasPruningCallback
from tqdm import tqdm

from dataClass3D import Data
from UNetDev3D import UNetDev
from trainUnet3D import DataSequence, LivePlotCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SilentDataSequence(DataSequence):
    """DataSequence that suppresses the 'Data loaded: X' print."""
    def __getitem__(self, idx):
        with suppress_stdout():
            return super().__getitem__(idx)


@contextlib.contextmanager
def suppress_stdout():
    """Suppress noisy stdout prints (e.g. 'Data loaded: X') during training."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


# ---------------------------------------------------------------------------
# Optuna study-level callbacks
# ---------------------------------------------------------------------------

class OptunaProgressCallback:
    """Prints a clear summary after every trial and saves best params to disk."""

    def __init__(self, save_path):
        self.save_path = save_path
        self.study_start = time.time()

    def __call__(self, study, trial):
        elapsed = time.time() - self.study_start
        finished = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        avg_str  = f"avg {format_duration(elapsed / finished)} / trial" if finished > 0 else "n/a"

        val_loss_str = f"{trial.value:.6e}" if trial.value is not None else "PRUNED/FAILED"

        lines = [
            "",
            "─" * 65,
            f"  Trial {trial.number + 1:>4}  │  State: {trial.state.name}",
            f"  Val loss : {val_loss_str}",
            f"  Params   : nCh={trial.params.get('nChannel','?')}, "
            f"deep={trial.params.get('deep','?')}, "
            f"gF={trial.params.get('growFactor','?')}, "
            f"fw={trial.params.get('frame_width','?')}, "
            f"opt={trial.params.get('optimizer','?')}, "
            f"lr={trial.params.get('learning_rate', 0):.2e}",
        ]

        try:
            best = study.best_trial
        except ValueError:
            best = None

        if best is not None:
            lines += [
                f"  ── Best so far ──────────────────────────────────────────",
                f"  Best loss: {best.value:.6e}  (trial #{best.number + 1})",
                f"  Best params: nCh={best.params.get('nChannel','?')}, "
                f"deep={best.params.get('deep','?')}, "
                f"gF={best.params.get('growFactor','?')}, "
                f"fw={best.params.get('frame_width','?')}, "
                f"opt={best.params.get('optimizer','?')}, "
                f"lr={best.params.get('learning_rate', 0):.2e}",
            ]
            best_params_file = self.save_path / 'best_hyperparameters.json'
            with open(best_params_file, 'w') as f:
                json.dump({'val_loss': best.value, 'trial': best.number + 1,
                           'params': best.params}, f, indent=4)

        lines += [
            f"  Progress : {finished} done, {pruned} pruned  │  "
            f"Elapsed: {format_duration(elapsed)}  │  {avg_str}",
            "─" * 65,
        ]
        tqdm.write("\n".join(lines))


class NoImprovementStopper:
    """Stops the study if best val_loss has not improved for `patience` completed trials."""

    def __init__(self, patience=15):
        self.patience = patience
        self._best = float('inf')
        self._no_improve = 0

    def __call__(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if study.best_value < self._best:
            self._best = study.best_value
            self._no_improve = 0
        else:
            self._no_improve += 1

        if self._no_improve >= self.patience:
            tqdm.write(f"\n  [Stopper] No improvement for {self.patience} consecutive trials. Stopping study.")
            study.stop()


# ---------------------------------------------------------------------------
# Keras epoch-level progress bar
# ---------------------------------------------------------------------------

class KerasTrialProgressCallback(keras.callbacks.Callback):
    """Single-line tqdm progress bar for epochs within a trial."""

    def __init__(self, trial_number, total_epochs):
        super().__init__()
        self.trial_number = trial_number
        self.total_epochs = total_epochs
        self._bar = None

    def on_train_begin(self, logs=None):
        self._bar = tqdm(
            total=self.total_epochs,
            desc=f"  Trial {self.trial_number}",
            unit="ep",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ep"
                       " [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # refresh=False prevents a separate redraw; update(1) does the single redraw
        self._bar.set_postfix(
            loss=f"{logs.get('loss', float('nan')):.3e}",
            val=f"{logs.get('val_loss', float('nan')):.3e}",
            refresh=False,
        )
        self._bar.update(1)

    def on_train_end(self, logs=None):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial, dataDirs, study_path, max_epochs, max_params):
    """Build, train, and evaluate one UNet3D configuration."""

    # Load data (suppress noisy prints)
    with suppress_stdout():
        data = Data(dataDirs)
    nx, ny, nz = data.nx, data.ny, data.nz
    dimIn, dimOut = data.dimIn, data.dimOut

    # Constrain deep so max-pooling never shrinks any spatial dim below 2
    max_deep = max(2, int(math.floor(math.log2(min(nx, ny, nz)))) - 1)

    # --- Hyperparameters ---
    nChannel       = trial.suggest_int('nChannel', 12, 18)
    deep           = trial.suggest_int('deep', 2, max_deep)
    growFactor     = trial.suggest_int('growFactor', 0, 3)
    frame_width    = trial.suggest_int('frame_width', 1, 6)
    learning_rate  = trial.suggest_float('learning_rate', 1.5e-4, 1.5e-4, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam'])  # extend as needed

    tqdm.write(f"\n  [Trial {trial.number + 1}] nCh={nChannel}, deep={deep}, "
               f"gF={growFactor}, fw={frame_width}, opt={optimizer_name}, lr={learning_rate:.2e}")

    trial_path = study_path / f"trial_{trial.number + 1:04d}"
    trial_params = {
        'trial': trial.number + 1,
        'nChannel': nChannel, 'deep': deep, 'growFactor': growFactor,
        'frame_width': frame_width, 'activation': 'relu',
        'optimizer': optimizer_name, 'learning_rate': learning_rate,
        'max_epochs': max_epochs, 'max_params': max_params,
    }

    # --- Build model ---
    net = UNetDev(
        n1=nx, n2=ny, n3=nz,
        dimIn=dimIn, dimOut=dimOut,
        frame_width=frame_width,
        nChannel=nChannel,
        deep=deep,
        growFactor=growFactor,
        scales=data.scales,
    )
    try:
        net.build()
    except Exception as e:
        tqdm.write(f"  [Trial {trial.number + 1}] Build failed: {e} → pruning.")
        raise optuna.TrialPruned()

    total_params = net.model.count_params()
    tqdm.write(f"  [Trial {trial.number + 1}] {total_params:,} parameters")
    trial_params['trainable_params'] = total_params

    if total_params > max_params:
        tqdm.write(f"  [Trial {trial.number + 1}] Exceeds budget "
                   f"({total_params:,} > {max_params:,}) → pruning.")
        raise optuna.TrialPruned()

    # --- Optimizer ---
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

    net.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    # --- Train / val split ---
    maxDataFiles = 3
    total_batches = data.nBatches
    n_val_batches   = max(maxDataFiles, int(total_batches * 0.1 // maxDataFiles) * maxDataFiles)
    n_train_batches = total_batches - n_val_batches

    train_seq = SilentDataSequence(data, maxDataFiles, startBatch=0,              nBatches=n_train_batches)
    val_seq   = SilentDataSequence(data, maxDataFiles, startBatch=n_train_batches, nBatches=n_val_batches)

    # Create output folder now so LivePlotCallback can save plots during training
    trial_path.mkdir(parents=True, exist_ok=True)

    # --- Callbacks ---
    progress_cb = KerasTrialProgressCallback(trial.number + 1, max_epochs)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        TFKerasPruningCallback(trial, 'val_loss'),
        progress_cb,
        LivePlotCallback(path=trial_path, plot_every=50),
    ]

    # --- Train (always close the progress bar, even on exception) ---
    try:
        history = net.model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=max_epochs,
            verbose=0,
            callbacks=callbacks,
        )
    except Exception:
        progress_cb.on_train_end()   # ensure bar is closed before re-raising
        raise

    val_loss   = min(history.history['val_loss'])
    epochs_run = len(history.history['val_loss'])

    if math.isnan(val_loss) or math.isinf(val_loss):
        tqdm.write(f"  [Trial {trial.number + 1}] NaN/Inf val_loss → pruning.")
        raise optuna.TrialPruned()

    # Save trial results
    trial_params.update({'epochs_run': epochs_run, 'val_loss': val_loss})
    with open(trial_path / 'trial_params.json', 'w') as f:
        json.dump(trial_params, f, indent=4)

    tqdm.write(f"  [Trial {trial.number + 1}] Done — {epochs_run} ep, "
               f"val_loss={val_loss:.6e}  → {trial_path.name}")

    tf.keras.backend.clear_session()
    return val_loss


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_optimization(dataDirs, path, max_epochs=500, max_params=10_000_000,
                     timeout_hours=None, patience=15, study_name='unet3d_optimization'):
    """Create/load an Optuna study and run optimization.

    Stopping criteria (whichever triggers first):
      - timeout_hours : wall-clock time limit (None = no limit)
      - patience      : stop after this many consecutive trials with no improvement
    """

    storage = f'sqlite:///{path / "optuna_unet3d.db"}'
    path.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=storage,
        load_if_exists=True,
    )

    already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if already_done > 0:
        print(f"\n  Resuming study '{study_name}' — {already_done} trials already completed.")

    print(f"  Stop when: no improvement for {patience} consecutive trials"
          + (f"  OR  timeout {timeout_hours}h" if timeout_hours else ""))
    print(f"  Max epochs per trial : {max_epochs}")
    print(f"  Max trainable params : {max_params:,}\n")

    N_MAX = 10_000
    timeout_sec = timeout_hours * 3600 if timeout_hours else None

    study.optimize(
        lambda trial: objective(trial, dataDirs, path, max_epochs, max_params),
        n_trials=N_MAX,
        timeout=timeout_sec,
        show_progress_bar=False,
        catch=(Exception,),
        callbacks=[OptunaProgressCallback(path), NoImprovementStopper(patience=patience)],
    )

    # Final summary
    print("\n" + "=" * 65)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 65)
    finished = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  Total: {len(study.trials)}  │  Completed: {finished}  │  Pruned: {pruned}")
    print(f"  Best val_loss: {study.best_trial.value:.6e}  (trial #{study.best_trial.number + 1})")
    print("\n  Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    best_params_file = path / 'best_hyperparameters.json'
    with open(best_params_file, 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)
    print(f"\n  Best parameters saved to: {best_params_file}")
    print("=" * 65)

    return study


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Suppress TensorFlow C++ / oneDNN noise that breaks tqdm rendering
    os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
    os.environ['XLA_FLAGS']              = '--xla_gpu_strict_conv_algorithm_picker=false'

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU device(s) found:")
        for device in physical_devices:
            print(f"  {device}")
    else:
        print("No GPU devices available.")

    print(f"\nPython  : {sys.version.split()[0]}")
    print(f"TF      : {tf.__version__}")
    print(f"Optuna  : {optuna.__version__}")

    # --- Configuration ---
    path = Path('"../../data/unet3D_optuna')
    dataDirs = [
        "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed_small/in15_vent10",
        "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed_small/in15_vent15",
        "../../reader3D/SimpleBladeExtrapolation/unsteady_interpolation/transformed_small/in15_vent20",
    ]

    MAX_EPOCHS = 100       # max epochs per trial (early stopping may cut short)
    MAX_PARAMS = 1_000_000 # max trainable parameters per trial
    PATIENCE   = 10        # stop study after N consecutive trials with no improvement
    TIMEOUT_H  = None      # optional wall-clock limit in hours (None = unlimited)

    run_optimization(dataDirs, path,
                     max_epochs=MAX_EPOCHS, max_params=MAX_PARAMS,
                     timeout_hours=TIMEOUT_H, patience=PATIENCE)

