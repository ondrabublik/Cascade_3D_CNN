"""Convert a trained UNet3D model (.keras) into one that has no fixed input size.

The network is fully convolutional, so the trained weights do not depend on the grid
size at all -- only the `InputLayer` stored inside the .keras file does.  This script
rewrites that declaration from

    batch_shape = [null, 48, 32, 16, 13]      ->      [null, null, null, null, 13]

and leaves `model.weights.h5` byte-for-byte untouched.  The resulting model is
numerically identical on the original grid and can additionally be evaluated on any
other grid (e.g. 96x64x32 or 192x128x64).

Usage
-----
    python convert_model_to_flexible.py ../data/test_coarse_unet_on_fine_mesh/model.keras
    python convert_model_to_flexible.py model.keras -o model_flex.keras
    python convert_model_to_flexible.py model.keras --verify            # needs tensorflow
    python convert_model_to_flexible.py model.keras --verify --test-shape 96 64 32
    python convert_model_to_flexible.py model.keras --info              # only print what is inside

With --rebuild the model is not patched but re-created from
UNetDev3D_one_param_flex.UNetDev and the weights are copied layer by layer
(equivalent result, useful if you want to change the architecture at the same time).
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path


# ----------------------------------------------------------------------
# reading information out of the .keras archive
# ----------------------------------------------------------------------
def loadConfig(path):
    with zipfile.ZipFile(path, "r") as z:
        return json.loads(z.read("config.json").decode("utf-8"))


def findLayers(cfg, className):
    return [l for l in cfg["config"]["layers"] if l["class_name"] == className]


def constantArgs(cfg, className):
    """All scalar (non-tensor) arguments of every op of the given class, in graph order."""
    values = []
    for layer in findLayers(cfg, className):
        for node in layer.get("inbound_nodes", []):
            for arg in node.get("args", []):
                if isinstance(arg, (int, float)) and not isinstance(arg, bool):
                    values.append(float(arg))
    return values


def readScales(cfg):
    """Recover the normalization constants baked into the graph.

    build() emits, in this order: (x - minVelMesh)/rangeVelMesh  (3x),
    (x - minVel)/rangeVel (4x), (x - minP)/rangeP (1x).
    """
    mins = constantArgs(cfg, "Subtract")        # these are the -min values
    ranges = constantArgs(cfg, "TrueDivide")    # these are max - min
    if len(mins) < 8 or len(ranges) < 8:
        raise RuntimeError("unexpected graph: cannot recover the scales")

    minVelMesh, rangeVelMesh = mins[0], ranges[0]
    minVel, rangeVel = mins[3], ranges[3]
    minP, rangeP = mins[7], ranges[7]

    return {
        "minVel": minVel,
        "maxVel": minVel + rangeVel,
        "minP": minP,
        "maxP": minP + rangeP,
        "minVelMesh": minVelMesh,
        "maxVelMesh": minVelMesh + rangeVelMesh,
    }


def readHyperParams(cfg):
    inputLayer = findLayers(cfg, "InputLayer")[0]["config"]
    shape = inputLayer.get("batch_shape") or inputLayer.get("input_shape")

    convs = [l["config"] for l in findLayers(cfg, "Conv3D")]
    nPool = len(findLayers(cfg, "MaxPooling3D"))

    deep = nPool + 1
    nChannel = convs[0]["filters"]
    kernel = convs[0]["kernel_size"][0]

    # growFactor: nChannels[i] = nChannel0 * i**growFactor for the encoder convs
    encoderFilters = [c["filters"] for c in convs[:nPool]]
    growFactor = 0 if all(f == nChannel for f in encoderFilters) else 1

    return {
        "gridShape": shape,
        "dimIn": shape[-1],
        "dimOut": convs[-1]["filters"],
        "deep": deep,
        "nChannel": nChannel,
        "frameWidth": (kernel - 1) // 2,
        "act": convs[0]["activation"],
        "actOut": convs[-1]["activation"],
        "growFactor": growFactor,
    }


# ----------------------------------------------------------------------
# the conversion itself
# ----------------------------------------------------------------------
def flexibleShape(shape):
    """[None, 48, 32, 16, 13] -> [None, None, None, None, 13]"""
    if not isinstance(shape, list) or len(shape) < 4:
        return shape
    if shape[0] is not None:
        return shape
    if not all(v is None or isinstance(v, int) for v in shape):
        return shape
    return [shape[0]] + [None] * (len(shape) - 2) + [shape[-1]]


def makeConfigFlexible(obj):
    """Recursively blank every spatial dimension in 'shape' / 'batch_shape' entries."""
    count = 0
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in ("shape", "batch_shape"):
                new = flexibleShape(val)
                if new != val:
                    obj[key] = new
                    count += 1
            else:
                count += makeConfigFlexible(val)
    elif isinstance(obj, list):
        for item in obj:
            count += makeConfigFlexible(item)
    return count


def convert(src, dst):
    src, dst = Path(src), Path(dst)
    with zipfile.ZipFile(src, "r") as z:
        names = z.namelist()
        cfg = json.loads(z.read("config.json").decode("utf-8"))
        payload = {n: z.read(n) for n in names if n != "config.json"}

    nChanged = makeConfigFlexible(cfg)
    payload["config.json"] = json.dumps(cfg).encode("utf-8")

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:                       # keep the original member order
            z.writestr(name, payload[name])

    return nChanged


def rebuild(src, dst):
    """Alternative route: build a fresh flexible model and copy the weights over."""
    import tensorflow as tf
    from UNetDev3D_one_param_flex import UNetDev

    cfg = loadConfig(src)
    hp = readHyperParams(cfg)
    scales = readScales(cfg)

    old = tf.keras.models.load_model(src, compile=False)

    net = UNetDev(
        None, None, None, hp["dimIn"], hp["dimOut"],
        act=hp["act"], actOut=hp["actOut"], scales=scales,
        frame_width=hp["frameWidth"], nChannel=hp["nChannel"],
        deep=hp["deep"], growFactor=hp["growFactor"],
    )
    net.build()

    oldConvs = [l for l in old.layers if isinstance(l, tf.keras.layers.Conv3D)]
    newConvs = [l for l in net.model.layers if isinstance(l, tf.keras.layers.Conv3D)]
    if len(oldConvs) != len(newConvs):
        raise RuntimeError("layer count mismatch: %d vs %d" % (len(oldConvs), len(newConvs)))

    for a, b in zip(oldConvs, newConvs):
        b.set_weights(a.get_weights())

    net.model.save(dst)
    print("rebuilt %d Conv3D layers" % len(newConvs))
    return net.model


# ----------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------
def verify(src, dst, testShape=None, seed=0):
    import numpy as np
    import tensorflow as tf

    old = tf.keras.models.load_model(src, compile=False)
    new = tf.keras.models.load_model(dst, compile=False)

    print("original input shape :", old.input_shape)
    print("converted input shape:", new.input_shape)

    shape = tuple(d for d in old.input_shape[1:4])
    dimIn = old.input_shape[-1]

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1,) + shape + (dimIn,), dtype=np.float32)
    # channel 6 is the boundary mask -> make it 0/1 so the BC branch is exercised
    x[..., 6] = (rng.random((1,) + shape) > 0.7).astype(np.float32)

    yOld = old.predict(x, verbose=0)
    yNew = new.predict(x, verbose=0)
    diff = float(np.max(np.abs(yOld - yNew)))
    print("max |old - new| on the original grid %s: %.3e" % (str(shape), diff))
    if diff != 0.0:
        print("WARNING: outputs are not bit-identical")
    else:
        print("OK: outputs are identical")

    if testShape:
        big = tuple(int(v) for v in testShape)
        xBig = rng.standard_normal((1,) + big + (dimIn,), dtype=np.float32)
        xBig[..., 6] = (rng.random((1,) + big) > 0.7).astype(np.float32)
        yBig = new.predict(xBig, verbose=0)
        print("converted model on %s -> output %s : OK" % (str(big), str(yBig.shape)))

    return diff


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to the trained .keras model")
    ap.add_argument("-o", "--output", default=None, help="output file (default: <name>_flex.keras)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild from UNetDev3D_one_param_flex instead of patching")
    ap.add_argument("--verify", action="store_true", help="compare the two models numerically (needs tensorflow)")
    ap.add_argument("--test-shape", nargs=3, type=int, default=None, metavar=("N1", "N2", "N3"),
                    help="also run the converted model on this grid, e.g. --test-shape 96 64 32")
    ap.add_argument("--info", action="store_true", help="only print what is stored in the model")
    args = ap.parse_args()

    src = Path(args.model)
    dst = Path(args.output) if args.output else src.with_name(src.stem + "_flex.keras")

    cfg = loadConfig(src)
    hp = readHyperParams(cfg)
    scales = readScales(cfg)
    print("model       :", src)
    print("input shape :", hp["gridShape"])
    print("hyperparams :", {k: v for k, v in hp.items() if k != "gridShape"})
    print("scales      :", json.dumps(scales))
    divisor = 2 ** (hp["deep"] - 1)
    print("every spatial dimension of the new model must be divisible by", divisor)

    if args.info:
        return

    if args.rebuild:
        rebuild(src, dst)
    else:
        n = convert(src, dst)
        print("patched %d shape entries -> %s" % (n, dst))

    if args.verify:
        verify(src, dst, testShape=args.test_shape)


if __name__ == "__main__":
    main()
