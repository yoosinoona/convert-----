import cv2
import sys

def stylize_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Lỗi: Không đọc được ảnh {input_path}")
        sys.exit(1)

    # HIỆU ỨNG SON DẦU / MÀU NƯỚC (Mượt mà, không nét viền đen)
    # sigma_s: độ mịn của màu (越大越平)
    # sigma_r: độ giữ lại chi tiết màu (越小色彩越flat)
    stylized = cv2.stylization(img, sigma_s=60, sigma_r=0.5)

    cv2.imwrite(output_path, stylized)
    print(f"Đã lưu ảnh stylized tại: {output_path}")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output_stylized.jpg"
    
    stylize_image(input_file, output_file)
