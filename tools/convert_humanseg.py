"""
Segmentation ONNX → NCNN
"""

import os
import sys
import subprocess
import shutil
import urllib.request

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"

NCNN_TOOLS_URLS = [
    "https://github.com/Tencent/ncnn/releases/download/20240410/ncnn-20240410-ubuntu-2204.zip",
    "https://github.com/Tencent/ncnn/releases/download/20230817/ncnn-20230817-ubuntu-2204.zip",
    "https://github.com/Tencent/ncnn/releases/download/20240410/ncnn-20240410-ubuntu.zip",
    "https://github.com/Tencent/ncnn/releases/download/20230817/ncnn-20230817-ubuntu.zip",
]

MODEL_SOURCES = [
    {
        "name": "U2Net salient (HuggingFace)",
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/u2net.onnx",
    },
    {
        "name": "RMBG-1.4 (HuggingFace)",
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx",
    },
]


def download_model():
    os.makedirs(WORK_DIR, exist_ok=True)
    onnx_path = os.path.join(WORK_DIR, "model.onnx")

    print("[1/3] Downloading segmentation ONNX model...")

    for source in MODEL_SOURCES:
        try:
            print(f"    Trying: {source['name']}...")
            req = urllib.request.Request(
                source["url"],
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                with open(onnx_path, "wb") as f:
                    shutil.copyfileobj(response, f)

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
    inp = model.graph.input[0]
    name = inp.name
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"    Input: {name} shape={shape}")
    print(f"    Outputs: {[o.name for o in model.graph.output]}")

    if all(s > 0 for s in shape):
        model_sim, check = simplify(model, input_shapes={name: shape})
        if check:
            onnx.save(model_sim, sim_path)
            print(f"    Simplified OK")
            return sim_path

    print("    Using original (skip simplify)")
    return onnx_path


def download_ncnn_tools():
    ncnn_dir = os.path.join(WORK_DIR, "ncnn-tools")
    if os.path.exists(ncnn_dir):
        return ncnn_dir

    print("[3/3] Downloading NCNN tools...")
    zip_path = os.path.join(WORK_DIR, "ncnn.zip")

    for url in NCNN_TOOLS_URLS:
        try:
            print(f"    Trying: {url.split('/')[-1]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                with open(zip_path, "wb") as f:
                    shutil.copyfileobj(response, f)
            print(f"    OK!")
            break
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    else:
        raise RuntimeError("All NCNN tools URLs failed")

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
    print("Segmentation ONNX → NCNN")
    print("=" * 50)

    os.makedirs(WORK_DIR, exist_ok=True)

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
