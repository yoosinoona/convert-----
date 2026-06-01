import cv2
import numpy as np
import sys

def style_transfer_onnx(input_path, model_path, output_path):
    # 1. Đọc ảnh đầu vào
    img = cv2.imread(input_path)
    if img is None:
        sys.exit(1)

    # 2. Chuẩn bị input cho ONNX (Cần định dạng NCHW, float32, normalize)
    inp = cv2.resize(img, (512, 512)) # Resize cho nhẹ và khớp model
    inp = inp.transpose((2, 0, 1)).astype(np.float32) # HWC -> CHW
    inp = np.expand_dims(inp, axis=0) # Thêm batch dimension -> 1, 3, 512, 512
    inp = inp / 255.0 # Normalize về 0-1

    # 3. Load model ONNX và chạy inference bằng OpenCV DNN
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setInput(inp)
    out = net.forward()

    # 4. Xử lý output
    out = out[0].transpose((1, 2, 0)) # CHW -> HWC
    out = np.clip(out * 255, 0, 255).astype(np.uint8) # Denormalize
    
    # Resize lại về kích thước gốc (nếu muốn)
    out = cv2.resize(out, (img.shape[1], img.shape[0]))

    cv2.imwrite(output_path, out)
    print(f"Đã lưu ảnh Neural Style tại: {output_path}")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output_neural.jpg"
    model_file = sys.argv[3] if len(sys.argv) > 3 else "model.onnx"
    
    style_transfer_onnx(input_file, model_file, output_file)
