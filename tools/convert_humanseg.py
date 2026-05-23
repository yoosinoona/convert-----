"""
Person Segmentation ONNX → NCNN
Dùng model ONNX có sẵn, không cần PaddlePaddle
"""

import os
import sys
import subprocess
import shutil
import urllib.request

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"

NCNN_TOOLS_URL = (
    "https://github.com/Tencent/ncnn/releases/download/20240820/"
    "ncnn-20240820-ubuntu.zip"
)

# Pre-exported ONNX models (thử lần lượt)
# ISNet general segmentation từ ONNX Model Zoo
MODEL_SOURCES = [
    {
        "name": "U2Net portrait (rembg)",
        "url": "https://github.com/danielgatis/rembg/raw/main/rembg/u2net/u2net.onnx",
    },
    {
        "name": "ISNet (ONNX Zoo)",
        "url": "https://github.com/danielgatis/rembg/raw/main/rembg/isnet-general-use/isnet-general-use.onnx",
    },
]


def run_cmd(cmd, cwd=None):
    print(f"    $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"    WARNING: exit code {result.returncode}")
    return result.returncode


def download_model():
    os.makedirs(WORK_DIR, exist_ok=True)
    onnx_path = os.path.join(WORK_DIR, "model.onnx")

    print("[1/3] Downloading person segmentation ONNX model...")

    for source in MODEL_SOURCES:
        try:
            print(f"    Trying: {source['name']}...")
            urllib.request.urlretrieve(source["url"], onnx_path)
            size_mb = os.path.getsize(onnx_path) / 1024 / 1024
            print(f"    OK! ({size_mb:.1f} MB)")
            return onnx_path
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    raise RuntimeError("All model sources failed")


def simplify_onnx(onnx_path):
    sim_path = os.path.join(WORK_DIR, "model_sim.onnx")
    print("[2/3] Simplifying ONNX...")

    import onnx
    from onnxsim import simplify

    model = onnx.load(onnx_path)
    input_info = model.graph.input[0]
    name = input_info.name
    shape = [d.dim_value for d in input_info.type.tensor_type.shape.dim]
    print(f"    Input: {name} shape={shape}")
    print(f"    Outputs: {[o.name for o in model.graph.output]}")

    # Only simplify if shape is valid
    if all(s > 0 for s in shape):
        model_sim, check = simplify(model, input_shapes={name: shape})
    else:
        print("    Skipping simplify (dynamic shape)")
        return onnx_path

    if check:
        onnx.save(model_sim, sim_path)
        print(f"    Simplified: {sim_path}")
        return sim_path
    else:
        print("    Simplify failed, using original")
        return onnx_path


def download_ncnn_tools():
    ncnn_dir = os.path.join(WORK_DIR, "ncnn-tools")
    if os.path.exists(ncnn_dir):
        return ncnn_dir

    print("[3/3] Downloading NCNN tools...")
    zip_path = os.path.join(WORK_DIR, "ncnn.zip")
    urllib.request.urlretrieve(NCNN_TOOLS_URL, zip_path)
    shutil.unpack_archive(zip_path, WORK_DIR)

    for d in os.listdir(WORK_DIR):
        full = os.path.join(WORK_DIR, d)
        if os.path.isdir(full) and d.startswith("ncnn-"):
            os.rename(full, ncnn_dir)
            break

    for tool in ["onnx2ncnn", "ncnn-optimize"]:
        p = os.path.join(ncnn_dir, "bin", tool)
        if os.path.exists(p):
            os.chmod(p, 0o755)

    return ncnn_dir


def convert_ncnn(model_path, ncnn_dir):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    onnx2ncnn = os.path.join(ncnn_dir, "bin", "onnx2ncnn")
    ncnn_opt = os.path.join(ncnn_dir, "bin", "ncnn-optimize")

    raw_p = os.path.join(WORK_DIR, "seg_raw.param")
    raw_b = os.path.join(WORK_DIR, "seg_raw.bin")

    print("[4/3] ONNX → NCNN...")
    subprocess.run([onnx2ncnn, model_path, raw_p, raw_b], check=True)

    print("[5/3] Optimizing...")
    opt_p = os.path.join(OUTPUT_DIR, "humansegv2.param")
    opt_b = os.path.join(OUTPUT_DIR, "humansegv2.bin")
    subprocess.run([ncnn_opt, raw_p, raw_b, opt_p, opt_b], check=True)

    ps = os.path.getsize(opt_p)
    bs = os.path.getsize(opt_b)
    print(f"    humansegv2.param ({ps:,} bytes)")
    print(f"    humansegv2.bin   ({bs:,} bytes)")
    print(f"    Total: {(ps + bs) / 1024 / 1024:.2f} MB")


def main():
    print("=" * 50)
    print("Person Segmentation ONNX → NCNN")
    print("=" * 50)

    onnx_path = download_model()
    sim_path = simplify_onnx(onnx_path)
    ncnn_dir = download_ncnn_tools()
    convert_ncnn(sim_path, ncnn_dir)

    print("=" * 50)
    print("DONE!")
    print(f"  {OUTPUT_DIR}/humansegv2.param")
    print(f"  {OUTPUT_DIR}/humansegv2.bin")
    print("=" * 50)

    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
