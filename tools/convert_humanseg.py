"""
Segmentation ONNX → NCNN via PNNX
Approach: Download ONNX → load bằng onnxruntime → export PyTorch → PNNX → NCNN
"""

import os
import sys
import shutil
import urllib.request
import numpy as np

WORK_DIR = "convert_tmp"
OUTPUT_DIR = "output"

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

    print("[1/5] Downloading segmentation ONNX model...")

    for source in MODEL_SOURCES:
        try:
            print(f"    Trying: {source['name']}...")
            req = urllib.request.Request(
                source["url"],
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                with open(onnx_path, "wb") as f:
                    shutil.copyfileobj(response, f)

            size_mb = os.path.getsize(onnx_path) / 1024 / 1024
            print(f"    OK! ({size_mb:.1f} MB)")
            return onnx_path

        except Exception as e:
            print(f"    Failed: {e}")
            continue

    raise RuntimeError("All model sources failed")


def convert_onnx_to_torch(onnx_path):
    """Load ONNX model, export weights as PyTorch model via onnx2torch"""
    print("[2/5] Converting ONNX → PyTorch...")

    try:
        # onnx2torch: convert ONNX model trực tiếp sang PyTorch
        from onnx2torch import convert
        torch_model = convert(onnx_path)
        torch_model.eval()
        print("    onnx2torch OK!")
        return torch_model
    except Exception as e:
        print(f"    onnx2torch failed: {e}")
        print("    Trying alternative approach...")
        return None


def convert_via_onnx_surgery(onnx_path):
    """
    Alternative: Load ONNX → trace qua PyTorch
    Dùng onnxruntime chạy inference, rồi build PyTorch wrapper
    """
    print("[2b/5] Alternative: PyTorch wrapper approach...")

    import torch
    import torch.nn as nn
    import onnxruntime as ort

    # Get model info
    sess = ort.InferenceSession(onnx_path)
    input_info = sess.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    # Resolve dynamic dims
    resolved_shape = []
    for d in input_shape:
        if isinstance(d, str) or d <= 0:
            resolved_shape.append(1)
        else:
            resolved_shape.append(d)

    print(f"    Input: {input_name} shape={resolved_shape}")

    # Create simple PyTorch wrapper
    class OnnxWrapper(nn.Module):
        def __init__(self, onnx_path, input_name, output_name):
            super().__init__()
            self.onnx_path = onnx_path
            self.input_name = input_name
            self.output_name = output_name
            self._sess = None

        def forward(self, x):
            if self._sess is None:
                self._sess = ort.InferenceSession(self.onnx_path)
            inp = {self.input_name: x.numpy()}
            out = self._sess.run([self.output_name], inp)
            return torch.from_numpy(out[0])

    output_name = sess.get_outputs()[0].name
    wrapper = OnnxWrapper(onnx_path, input_name, output_name)
    wrapper.eval()
    print(f"    Wrapper created (input={input_name}, output={output_name})")
    return wrapper, resolved_shape


def convert_to_ncnn(torch_model, input_shape):
    """Export PyTorch model → NCNN via PNNX"""
    import torch
    import pnnx

    print("[3/5] Converting PyTorch → NCNN via PNNX...")

    dummy = torch.randn(*input_shape)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pnnx.export(torch_model, "seg", inputs=dummy)
    print("    PNNX export done!")

    # Move output files
    for ext in [".ncnn.param", ".ncnn.bin"]:
        src = f"seg{ext}"
        dst = os.path.join(OUTPUT_DIR, f"humansegv2{ext}")
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"    {dst} ({os.path.getsize(dst):,} bytes)")

    # Cleanup
    for f in os.listdir("."):
        if f.startswith("seg.") and f.endswith((".param", ".bin", ".py")):
            os.remove(f)

    return True


def verify():
    print("[4/5] Verifying...")
    param = os.path.join(OUTPUT_DIR, "humansegv2.ncnn.param")
    binf = os.path.join(OUTPUT_DIR, "humansegv2.ncnn.bin")

    if os.path.exists(param) and os.path.exists(binf):
        ps = os.path.getsize(param) / 1024
        bs = os.path.getsize(binf) / 1024
        print(f"    humansegv2.ncnn.param: {ps:.1f} KB")
        print(f"    humansegv2.ncnn.bin:   {bs:.1f} KB")
        print(f"    Total: {(ps + bs) / 1024:.1f} MB")

        with open(param, "r") as f:
            content = f.read()
            if "Input" in content:
                print("    Param file OK")
            else:
                print("    WARNING: no Input layer found")

        return True
    else:
        print(f"    FAILED! Files: {os.listdir(OUTPUT_DIR)}")
        return False


def main():
    print("=" * 50)
    print("Segmentation ONNX → NCNN (via PNNX)")
    print("=" * 50)

    os.makedirs(WORK_DIR, exist_ok=True)

    # Step 1: Download ONNX
    onnx_path = download_model()

    # Step 2: Try onnx2torch first
    torch_model = convert_onnx_to_torch(onnx_path)

    if torch_model is not None:
        # Determine input shape
        import onnx
        model = onnx.load(onnx_path)
        shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        input_shape = [1 if (isinstance(s, str) or s <= 0) else s for s in shape]
        if len(input_shape) != 4:
            input_shape = [1, 3, 192, 192]
    else:
        # Fallback: wrapper approach
        torch_model, input_shape = convert_via_onnx_surgery(onnx_path)

    print(f"    Input shape: {input_shape}")

    # Step 3: PNNX → NCNN
    convert_to_ncnn(torch_model, input_shape)

    # Step 4: Verify
    ok = verify()

    print("=" * 50)
    if ok:
        print("DONE!")
    else:
        print("FAILED!")
        sys.exit(1)
    print("=" * 50)

    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
