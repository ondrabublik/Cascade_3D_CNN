import json
import sys
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported. See trainUnet3D_multistep.py.
matplotlib.use("Agg")

import tensorflow as tf
import tensorflow.keras as keras
from keras.optimizers import Adam

from trainUnet3D_multistep import (
    ErrsEqs,
    LivePlotCallback,
    MultiStepDataSequence,
    build_multistep_rollout_model,
    plotErrs,
    plotLoss,
    weighted_uvwp_mse,
)


def trainNetMultistep(
    unet,
    path,
    modelPath,
    dataDirs=None,
    epochs=100,
    batch_size=2,
    learningRate=1e-4,
    act="relu",
    actOut="sigmoid",
    frameWidth=2,
    nChannel=16,
    deep=5,
    growFactor=1,
    n_steps=5,
    velocityLossWeight=2.0,
    pressureLossWeight=0.5,
    validationSplit=0.05,
    clipNorm=1.0,
):
    data = Data(dataDirs)
    nx, ny, nz = data.nx, data.ny, data.nz
    dimIn = data.dimIn
    dimOut = data.dimOut
    print(f"nx = {nx}, ny = {ny}, nz = {nz}, n_steps = {n_steps}")

    net = unet(
        nx,
        ny,
        nz,
        dimIn,
        dimOut,
        act=act,
        actOut=actOut,
        scales=data.scales,
        frame_width=frameWidth,
        nChannel=nChannel,
        deep=deep,
        growFactor=growFactor,
    )

    modelPath = Path(modelPath)
    if not modelPath.exists():
        sys.exit("Error: model file does not exist: " + str(modelPath))

    print(f"Loading model from {modelPath}")
    # Load the saved base UNet (not the multistep wrapper). Architecture comes from
    # the checkpoint; nChannel/deep/frameWidth here must match that saved model.
    net.model = keras.models.load_model(
        modelPath,
        safe_mode=False,
        compile=False,
        custom_objects={"slice": slice, "tf": tf},
    )
    net.info()
    base_model = net.model

    # Ensure multistep datasets exist (step axis is the first axis after batch).
    first_multistep_file = data.dataPath / Path("dataIn_multistep_0.npy")
    if not first_multistep_file.exists():
        data.prepare_training_data_multistep(nSteps=n_steps)

    multistep_model = build_multistep_rollout_model(base_model, n_steps=n_steps)
    # clipnorm guards the rollout: the gradient runs through n_steps chained calls of
    # the same UNet, where one bad sequence can otherwise produce a huge update.
    optimizer = Adam(learning_rate=learningRate, clipnorm=clipNorm)
    multistep_model.compile(
        loss=weighted_uvwp_mse(
            velocity_weight=velocityLossWeight,
            pressure_weight=pressureLossWeight,
        ),
        optimizer=optimizer,
    )

    path.mkdir(parents=True, exist_ok=True)
    input_params = {
        "modelName": str(path.name),
        "UnetName": str(net.name),
        "dataDirectory": str(dataDirs),
        "modelPath": str(modelPath),
        "epochs": epochs,
        "batch_size": batch_size,
        "learningRate": learningRate,
        "actOut": actOut,
        "frameWidth": frameWidth,
        "nChannel": nChannel,
        "deep": deep,
        "nSteps": n_steps,
        "velocityLossWeight": velocityLossWeight,
        "pressureLossWeight": pressureLossWeight,
        "validationSplit": validationSplit,
        "clipNorm": clipNorm,
    }
    file_path = path / Path("train_params.json")
    with file_path.open("w") as file:
        json.dump(input_params, fp=file, indent=4)

    # Hold the last files back for validation. fit() cannot do validation_split on a
    # Sequence -- it only splits arrays it can slice -- so the split has to be explicit.
    nVal = int(round(data.nBatches * validationSplit))
    nVal = min(nVal, data.nBatches - 1)
    trainIds = range(data.nBatches - nVal)
    valIds = range(data.nBatches - nVal, data.nBatches)

    train_data_sequence = MultiStepDataSequence(data, trainIds)
    val_data_sequence = MultiStepDataSequence(data, valIds) if nVal else None
    print(f"train files: {len(trainIds)}, validation files: {nVal}")

    # Without a validation split there is no val_loss to watch, so fall back to loss.
    monitorKey = "val_loss" if val_data_sequence else "loss"
    print(f"monitoring '{monitorKey}' for the best checkpoint")

    live_plot = LivePlotCallback(path=path, plot_every=10)

    history = multistep_model.fit(
        train_data_sequence,
        validation_data=val_data_sequence,
        shuffle=True,
        epochs=epochs,
        verbose=1,
        callbacks=[ErrsEqs(net, path, monitor=monitorKey), live_plot],
    )

    # Save only base UNet model (without multistep Lambda wrapper).
    base_model.save(path / Path("model.keras"))
    return history


if __name__ == "__main__":
    from UNetDev3D_one_param_flex import UNetDev as Unet
    from dataClass3D_one_param import Data

    import os

    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        print("No GPU devices available.")
    else:
        print("GPU device(s) found:")
        for device in physical_devices:
            print(f"  {device}")

    print("\nPython version: " + sys.version.split()[0])
    print("TensorFlow version: " + tf.__version__)

    # Anchor paths to this file, not to the current working directory.
    projectDir = Path(__file__).resolve().parents[1]

    dataDirs = [
        str(projectDir.parent / "reader3D" / "FinalBladeCascade" / "data"/ "transformed_10o"),
        str(projectDir.parent / "reader3D" / "FinalBladeCascade" / "data" / "transformed_15o"),
        str(projectDir.parent / "reader3D" / "FinalBladeCascade" / "data" / "transformed_20o")
    ]

    path = projectDir / "data" / "net7_3D_multistep_lowo_v4"
    modelPath = path / "model.keras"

    hist = trainNetMultistep(
        unet=Unet,
        dataDirs=dataDirs,
        epochs=20000,
        batch_size=3,
        frameWidth=2,
        nChannel=26,
        deep=7,
        growFactor=0,
        learningRate=1e-4,
        path=path,
        modelPath=modelPath,
        n_steps=5,
        velocityLossWeight=1.0,
        pressureLossWeight=0.1,
        validationSplit=0.05,
    )

    plotLoss(history=hist, path=path)
    plotErrs(path=path)
