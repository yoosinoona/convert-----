"""
Convert PP-HumanSegV2 Mobile → NCNN
Pipeline: PaddlePaddle → ONNX → Simplify → NCNN → Optimize
"""

import os
import subprocess
import shutil
import urllib.request
import tarfile

# ─── Config ────────────────────────────────────────

MODEL_URL = (
    "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/"
    "pp_humansegv2_mobile_192x192_inference_model.tar"
)
NCNN_TOOLS_URL = (
    "https://github.com/Tencent/ncnn/releases/download/20240820/"
    "ncnn-20240820-ubuntu.zip"
)

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"
MODEL_DIR = os.path.join(WORK_DIR, "pp_humansegv2_mobile_192x192_inference_model")

PARAMS = {
    "model_dir": MODEL_DIR,
    "model_filename": "model.pdmodel",
    "params_filename": "model.pdiparams",
    "opset_version": 11,
}

# ─── Step 1: Download ──────────────────────────────

def download_model():
    os.makedirs(WORK_DIR, exist_ok=True)
    tar_path = os.path.join(WORK_DIR, "model.tar")
    if not os.path.exists(tar_path):
        print("[1/6] Downloading PP-HumanSegV2 model...")
        urllib.request.urlretrieve(MODEL_URL, tar_path)
    else:
        print("[1/6] Model already downloaded.")

    print("[2/6] Extracting...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=WORK_DIR)
    print(f"    Model dir: {MODEL_DIR}")


# ─── Step 2: PaddlePaddle → ONNX ───────────────────

def paddle_to_onnx():
    onnx_path = os.path.join(WORK_DIR, "humansegv2.onnx")
    print("[3/6] Converting PaddlePaddle → ONNX...")

    import paddle2onnx
    paddle2onnx.command.run(
        model_dir=PARAMS["model_dir"],
        model_filename=PARAMS["model_filename"],
        params_filename=PARAMS["params_filename"],
        save_file=onnx_path,
        opset_version=PARAMS["opset_version"],
        enable_onnx_checker=True,
    )
    print(f"    Saved: {onnx_path}")
    return onnx_path


# ─── Step 3: Simplify ONNX ─────────────────────────

def simplify_onnx(onnx_path):
    sim_path = os.path.join(WORK_DIR, "humansegv2_sim.onnx")
    print("[4/6] Simplifying ONNX...")

    import onnx
    from onnxsim import simplify

    model = onnx.load(onnx_path)
    model_sim, check = simplify(model, input_shapes={"x": [1, 3, 192, 192]})
    assert check, "ONNX simplify failed!"
    onnx.save(model_sim, sim_path)

    # Print model info
    print(f"    Inputs:  {[i.name for i in model_sim.graph.input]}")
    print(f"    Outputs: {[o.name for o in model_sim.graph.output]}")
    print(f"    Saved:   {sim_path}")
    return sim_path


# ─── Step 4: Download NCNN tools ───────────────────

def download_ncnn_tools():
    ncnn_dir = os.path.join(WORK_DIR, "ncnn-tools")
    if os.path.exists(ncnn_dir):
        print("[5/6] NCNN tools already exist.")
        return ncnn_dir

    print("[5/6] Downloading NCNN tools...")
    zip_path = os.path.join(WORK_DIR, "ncnn.zip")
    urllib.request.urlretrieve(NCNN_TOOLS_URL, zip_path)
    shutil.unpack_archive(zip_path, WORK_DIR)

    # Rename to consistent path
    extracted = None
    for d in os.listdir(WORK_DIR):
        full = os.path.join(WORK_DIR, d)
        if os.path.isdir(full) and d.startswith("ncnn-"):
            extracted = full
            break

    os.rename(extracted, ncnn_dir)
    os.chmod(os.path.join(ncnn_dir, "bin", "onnx2ncnn"), 0o755)
    os.chmod(os.path.join(ncnn_dir, "bin", "ncnn-optimize"), 0o755)
    print(f"    Tools at: {ncnn_dir}/bin/")
    return ncnn_dir


# ─── Step 5: ONNX → NCNN ──────────────────────────

def onnx_to_ncnn(sim_path, ncnn_tools_dir):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    onnx2ncnn = os.path.join(ncnn_tools_dir, "bin", "onnx2ncnn")
    ncnn_opt = os.path.join(ncnn_tools_dir, "bin", "ncnn-optimize")

    raw_param = os.path.join(WORK_DIR, "humansegv2_raw.param")
    raw_bin = os.path.join(WORK_DIR, "humansegv2_raw.bin")

    print("[6a/6] Converting ONNX → NCNN...")
    subprocess.run(
        [onnx2ncnn, sim_path, raw_param, raw_bin],
        check=True,
    )

    print("[6b/6] Optimizing NCNN model...")
    opt_param = os.path.join(OUTPUT_DIR, "humansegv2.param")
    opt_bin = os.path.join(OUTPUT_DIR, "humansegv2.bin")
    subprocess.run(
        [ncnn_opt, raw_param, raw_bin, opt_param, opt_bin],
        check=True,
    )

    # Stats
    p_size = os.path.getsize(opt_param)
    b_size = os.path.getsize(opt_bin)
    print(f"    Output: {opt_param} ({p_size:,} bytes)")
    print(f"    Output: {opt_bin} ({b_size:,} bytes)")
    print(f"    Total:  {(p_size + b_size) / 1024 / 1024:.2f} MB")


# ─── Main ──────────────────────────────────────────

def main():
    print("=" * 50)
    print("PP-HumanSegV2 → NCNN Converter")
    print("=" * 50)

    download_model()
    onnx_path = paddle_to_onnx()
    sim_path = simplify_onnx(onnx_path)
    ncnn_tools_dir = download_ncnn_tools()
    onnx_to_ncnn(sim_path, ncnn_tools_dir)

    print()
    print("=" * 50)
    print("DONE! Files in output/:")
    print(f"  - humansegv2.param")
    print(f"  - humansegv2.bin")
    print("=" * 50)

    # Cleanup
    print("Cleaning up...")
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
