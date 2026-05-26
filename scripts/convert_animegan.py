import os
import sys
import subprocess
import shutil


def find_onnx2ncnn():
    """Find onnx2ncnn binary"""
    # Check PATH
    path = shutil.which("onnx2ncnn")
    if path:
        return path
    # Fallback: local ncnn release
    for candidate in [
        "ncnn-20240820-ubuntu-2204/bin/onnx2ncnn",
        "ncnn-20230820-ubuntu-2204/bin/onnx2ncnn",
    ]:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    print("ERROR: onnx2ncnn not found!")
    sys.exit(1)


def main():
    print("=== AnimeGANv3 ONNX -> NCNN (via onnx2ncnn) ===\n")

    onnx2ncnn = find_onnx2ncnn()
    print(f"onnx2ncnn: {onnx2ncnn}")

    onnx_file = "AnimeGANv3_PortraitSketch_25.onnx"
    if not os.path.exists(onnx_file):
        print(f"MISSING: {onnx_file}")
        sys.exit(1)

    print(f"Input: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    # 1. Simplify ONNX
    print("\n1. Simplifying ONNX...")
    sim_file = "animegan_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"   Simplify failed, using original: {ret.stderr[:200]}")
        shutil.copy(onnx_file, sim_file)
    else:
        print(f"   OK: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 2. Convert ONNX -> NCNN via onnx2ncnn binary
    print("\n2. Converting via onnx2ncnn...")
    out_param = "animegan.param"
    out_bin = "animegan.bin"

    for f in [out_param, out_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        [onnx2ncnn, sim_file, out_param, out_bin],
        capture_output=True, text=True, timeout=300,
    )
    print(f"   stdout (last 500): {ret.stdout[-500:]}")
    if ret.stderr:
        print(f"   stderr (last 500): {ret.stderr[-500:]}")

    if not os.path.exists(out_param) or not os.path.exists(out_bin):
        print("   onnx2ncnn output files not found!")
        sys.exit(1)

    print(f"   OK: param={os.path.getsize(out_param) / 1024:.1f} KB, "
          f"bin={os.path.getsize(out_bin) / 1024 / 1024:.1f} MB")

    # 3. Copy to output
    os.makedirs("output", exist_ok=True)
    shutil.copy(out_param, "output/animegan.param")
    shutil.copy(out_bin, "output/animegan.bin")

    # 4. Verify
    print("\n=== Output ===")
    for f in ["output/animegan.param", "output/animegan.bin"]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  {f}: {size / 1024 / 1024:.1f} MB"
                  if size > 1024 * 1024
                  else f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
