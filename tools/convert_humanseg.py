"""
Convert PP-HumanSegV2 Mobile → NCNN
Pipeline: PaddleSeg → ONNX → Simplify → NCNN → Optimize
"""

import os
import subprocess
import shutil
import urllib.request

# ─── Config ────────────────────────────────────────

NCNN_TOOLS_URL = (
    "https://github.com/Tencent/ncnn/releases/download/20240820/"
    "ncnn-20240820-ubuntu.zip"
)

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"


# ─── Step 1: Export PaddleSeg model to ONNX ────────

def export_to_onnx():
    os.makedirs(WORK_DIR, exist_ok=True)
    onnx_path = os.path.join(WORK_DIR, "humansegv2.onnx")

    print("[1/4] Exporting PP-HumanSegV2 via paddleseg...")

    import paddle
    from paddleseg.models import PPMobileSeg

    # Load pretrained model
    model = PPMobileSeg(
        num_classes=2,
        backbone_channels=[32, 64, 128],
        head_channels=128,
        align_corners=False,
    )

    # Download pretrained weights
    model_url = "https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/pp_humansegv2_mobile_192x192_pretrained/model.pdparams"
    weights_path = os.path.join(WORK_DIR, "model.pdparams")

    if not os.path.exists(weights_path):
        print("    Downloading weights...")
        urllib.request.urlretrieve(model_url, weights_path)

    model.set_state_dict(paddle.load(weights_path))
    model.eval()

    # Export to ONNX
    print("    Converting to ONNX...")
    input_spec = paddle.static.InputSpec(
        shape=[1, 3, 192, 192], dtype="float32"
    )

    paddle.onnx.export(
        model,
        os.path.join(WORK_DIR, "humansegv2"),
        input_spec=[input_spec],
        opset_version=11,
    )

    # Rename output
    exported = os.path.join(WORK_DIR, "humansegv2.onnx")
    if os.path.exists(exported):
        print(f"    Saved: {exported}")
        return exported

    raise FileNotFoundError("ONNX export failed")


# ─── Step 2: Simplify ONNX ─────────────────────────

def simplify_onnx(onnx_path):
    sim_path = os.path.join(WORK_DIR, "humansegv2_sim.onnx")
    print("[2/4] Simplifying ONNX...")

    import onnx
    from onnxsim import simplify

    model = onnx.load(onnx_path)
    model_sim, check = simplify(
        model,
        input_shapes={"x": [1, 3, 192, 192]}
    )
    assert check, "ONNX simplify failed!"
    onnx.save(model_sim, sim_path)

    print(f"    Inputs:  {[i.name for i in model_sim.graph.input]}")
    print(f"    Outputs: {[o.name for o in model_sim.graph.output]}")
    print(f"    Saved:   {sim_path}")
    return sim_path


# ─── Step 3: Download NCNN tools ───────────────────

def download_ncnn_tools():
    ncnn_dir = os.path.join(WORK_DIR, "ncnn-tools")
    if os.path.exists(ncnn_dir):
        print("[3/4] NCNN tools already exist.")
        return ncnn_dir

    print("[3/4] Downloading NCNN tools...")
    zip_path = os.path.join(WORK_DIR, "ncnn.zip")
    urllib.request.urlretrieve(NCNN_TOOLS_URL, zip_path)
    shutil.unpack_archive(zip_path, WORK_DIR)

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


# ─── Step 4: ONNX → NCNN ──────────────────────────

def onnx_to_ncnn(sim_path, ncnn_tools_dir):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    onnx2ncnn = os.path.join(ncnn_tools_dir, "bin", "onnx2ncnn")
    ncnn_opt = os.path.join(ncnn_tools_dir, "bin", "ncnn-optimize")

    raw_param = os.path.join(WORK_DIR, "humansegv2_raw.param")
    raw_bin = os.path.join(WORK_DIR, "humansegv2_raw.bin")

    print("[4a/4] Converting ONNX → NCNN...")
    subprocess.run([onnx2ncnn, sim_path, raw_param, raw_bin], check=True)

    print("[4b/4] Optimizing NCNN model...")
    opt_param = os.path.join(OUTPUT_DIR, "humansegv2.param")
    opt_bin = os.path.join(OUTPUT_DIR, "humansegv2.bin")
    subprocess.run([ncnn_opt, raw_param, raw_bin, opt_param, opt_bin], check=True)

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

    onnx_path = export_to_onnx()
    sim_path = simplify_onnx(onnx_path)
    ncnn_tools_dir = download_ncnn_tools()
    onnx_to_ncnn(sim_path, ncnn_tools_dir)

    print()
    print("=" * 50)
    print("DONE! Files in output/:")
    print("  - humansegv2.param")
    print("  - humansegv2.bin")
    print("=" * 50)

    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
