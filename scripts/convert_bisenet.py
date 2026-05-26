import torch
import torch.nn as nn
import os
import sys
import shutil
import subprocess
import gc


class BiSeNetWrapper(nn.Module):
    """Wrapper to return only main output (drop aux outputs)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def find_tool(name):
    """Find ncnn tool binary"""
    path = shutil.which(name)
    if path:
        return path
    for candidate in [
        f"ncnn-20240820-ubuntu-2204/bin/{name}",
        f"ncnn-20230820-ubuntu-2204/bin/{name}",
    ]:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    print(f"ERROR: {name} not found!")
    sys.exit(1)


def run_cmd(cmd, timeout=300):
    """Run subprocess and print result"""
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if ret.stdout:
        print(f"   stdout: {ret.stdout[-300:]}")
    if ret.stderr:
        print(f"   stderr: {ret.stderr[-300:]}")
    return ret


def optimize_fp16(ncnnoptimize, in_param, in_bin, out_param, out_bin):
    """Convert FP32 ncnn model to FP16 using ncnnoptimize
       flag: 0 = fp32, 65536 = fp16
    """
    for f in [out_param, out_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = run_cmd([ncnnoptimize, in_param, in_bin, out_param, out_bin, "65536"])

    if not os.path.exists(out_param) or not os.path.exists(out_bin):
        print(f"   FP16 output not found!")
        return False
    return True


def main():
    print("=== BiSeNet -> NCNN (FP32 + FP16) ===\n")

    onnx2ncnn = find_tool("onnx2ncnn")
    ncnnoptimize = find_tool("ncnnoptimize")
    print(f"onnx2ncnn:    {onnx2ncnn}")
    print(f"ncnnoptimize: {ncnnoptimize}")

    sys.path.insert(0, 'repo_bisenet')
    from model import BiSeNet

    # 1. Load model
    model = BiSeNet(n_classes=19)
    weight_path = "repo_bisenet/79999_iter.pth"
    state_dict = torch.load(weight_path, map_location='cpu')
    clean = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        clean[name] = v
    model.load_state_dict(clean, strict=False)
    model.eval()
    print("Weights loaded!")

    # 2. Wrap
    wrapper = BiSeNetWrapper(model)
    wrapper.eval()

    # 3. Export to ONNX
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    onnx_file = "bisenet.onnx"
    print("\nExporting to ONNX...")
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"  Forward: {list(dummy.shape)} -> {list(out.shape)}")

    torch.onnx.export(
        wrapper, dummy, onnx_file,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes=None,
    )
    print(f"  ONNX exported: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    del wrapper, model, dummy, out
    gc.collect()

    # 4. Simplify ONNX
    print("\nSimplifying ONNX...")
    sim_file = "bisenet_sim.onnx"
    ret = run_cmd([sys.executable, "-m", "onnxsim", onnx_file, sim_file])
    if ret.returncode != 0:
        print(f"   Simplify failed, using original")
        shutil.copy(onnx_file, sim_file)
    else:
        print(f"   OK: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 5. Convert ONNX -> NCNN FP32
    print("\nConverting to FP32 via onnx2ncnn...")
    fp32_param = "bisenet_fp32.param"
    fp32_bin = "bisenet_fp32.bin"

    for f in [fp32_param, fp32_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = run_cmd([onnx2ncnn, sim_file, fp32_param, fp32_bin])

    if not os.path.exists(fp32_param) or not os.path.exists(fp32_bin):
        print("   onnx2ncnn output files not found!")
        sys.exit(1)

    fp32_size = os.path.getsize(fp32_bin)
    print(f"   FP32 OK: param={os.path.getsize(fp32_param) / 1024:.1f} KB, "
          f"bin={fp32_size / 1024 / 1024:.1f} MB")

    # 6. Convert FP32 -> FP16
    print("\nConverting to FP16 via ncnnoptimize...")
    fp16_param = "bisenet_fp16.param"
    fp16_bin = "bisenet_fp16.bin"

    if optimize_fp16(ncnnoptimize, fp32_param, fp32_bin, fp16_param, fp16_bin):
        fp16_size = os.path.getsize(fp16_bin)
        print(f"   FP16 OK: param={os.path.getsize(fp16_param) / 1024:.1f} KB, "
              f"bin={fp16_size / 1024 / 1024:.1f} MB")
        print(f"   Size reduction: {(1 - fp16_size / fp32_size) * 100:.1f}%")
    else:
        print("   FP16 conversion failed, skipping FP16 output")
        fp16_param = None
        fp16_bin = None

    # 7. Copy to output
    os.makedirs("output", exist_ok=True)
    shutil.copy(fp32_param, "output/biSeNet.param")
    shutil.copy(fp32_bin, "output/biSeNet.bin")

    if fp16_param and fp16_bin:
        shutil.copy(fp16_param, "output/biSeNet_fp16.param")
        shutil.copy(fp16_bin, "output/biSeNet_fp16.bin")

    # 8. Verify
    print("\n=== Output ===")
    output_files = [
        "output/biSeNet.param",
        "output/biSeNet.bin",
        "output/biSeNet_fp16.param",
        "output/biSeNet_fp16.bin",
    ]
    for f in output_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            if size > 1024 * 1024:
                print(f"  {f}: {size / 1024 / 1024:.1f} MB")
            else:
                print(f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  {f}: (not generated)")

    print("\nBiSeNet OK!")


if __name__ == "__main__":
    main()
