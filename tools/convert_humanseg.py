"""
Person Segmentation ONNX → NCNN
"""

import os
import sys
import subprocess
import shutil
import urllib.request

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"

NCNN_TOOLS_URLS = [
    # Thử nhiều format khác nhau
    "https://github.com/Tencent/ncnn/releases/download/20240410/ncnn-20240410-ubuntu-2204.zip",
    "https://github.com/Tencent/ncnn/releases/download/20230817/ncnn-20230817-ubuntu-2204.zip",
    "https://github.com/Tencent/ncnn/releases/download/20240410/ncnn-20240410-ubuntu.zip",
    "https://github.com/Tencent/ncnn/releases/download/20230817/ncnn-20230817-ubuntu.zip",
]


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
            with urllib.request.urlopen(req) as response:
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
    print("Segmentation ONNX → NCNN")
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
