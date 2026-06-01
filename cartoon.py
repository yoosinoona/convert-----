import cv2
import sys
import os

def cartoonify(input_path, output_path):
    # 1. Đọc ảnh
    img = cv2.imread(input_path)
    if img is None:
        print(f"Lỗi: Không đọc được ảnh {input_path}")
        sys.exit(1)

    # 2. LÀM PHẲNG MÀU SẮC (Color Smoothing)
    color = img.copy()
    for _ in range(7):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. TRÍCH XUẤT ĐƯỜNG NÉT ĐEN (Edge Detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 
                                  blockSize=9, C=2)

    # 4. KẾT HỢP MÀU VÀ NÉT VẼ
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges)

    # 5. Lưu ảnh kết quả
    cv2.imwrite(output_path, cartoon)
    print(f"Đã lưu ảnh cartoon tại: {output_path}")

if __name__ == "__main__":
    # Lấy tham số từ command line hoặc dùng giá trị mặc định
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output_cartoon.jpg"
    
    cartoonify(input_file, output_file)
