## Stage A — architecture study (`--study arch`)

| parameter    | range          | note                             |
|--------------|----------------|----------------------------------|
| `nChannel`   | 8 … 32, step 2 | base channel count               |
| `frameWidth` | 1, 2, 3        | Conv3D kernel = 2·frameWidth + 1 |
| `growFactor` | 0, 1           | constant vs. growing channels    |
| `act`        | relu, elu, swish | hidden activation              |

Training parameters fixed at the article defaults: `lr=1e-4`, `clipNorm=1.0`,
loss weights 1.0/0.1. Output activation stays `sigmoid` (the built-in
denormalization requires outputs in [0, 1]).

## Stage B — training study (`--study train`)

| parameter            | range                    | note                        |
|----------------------|--------------------------|-----------------------------|
| `learningRate`       | 1e-5 … 3e-3, log-uniform | Adam                        |
| `clipNorm`           | 0.1 … 10, log-uniform    | gradient clipping           |

Loss weights are fixed at the article values (velocity 1.0, pressure 0.1).

The architecture is fixed: **automatically taken from the best trial of the
finished arch study**; if none exists, the article defaults are used
(`nChannel=26, frameWidth=2, growFactor=0, act=relu`). Override with
`--nChannel/--frameWidth/--growFactor/--act`.

## Objective (both stages)

Best value of a **fixed reference metric** `val_ref_loss` — the weighted u/v/w/p
validation MSE with the article weights (velocity 1.0, pressure 0.1), computed
as a Keras metric independently of the trained loss. With the loss weights
fixed everywhere it equals `val_loss`; it is kept as a separate metric so the
objective definition never changes if loss weights are searched again later.

Common fixed settings: `deep=5`, `n_steps=5`, `batch_size=2`, validation split
0.2 (last 20 % of batch files), seed 42.

## Run

```
..\.venv\Scripts\python optuna_search.py --study arch  --n-trials 100 --epochs 300
..\.venv\Scripts\python optuna_search.py --study train --n-trials 100 --epochs 300
```

- Run **arch first, then train** (train reads arch's best architecture from the
  database).

## Tables for the article

```
..\.venv\Scripts\python optuna_make_table.py --study arch  --top 10
..\.venv\Scripts\python optuna_make_table.py --study train --top 10
```

## Optuna dashboard

```
../.venv/Scripts/optuna-dashboard sqlite:///optuna_unet3d.db
```