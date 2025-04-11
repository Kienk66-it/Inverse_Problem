# Bài Tập Môn Các Phương Pháp Xử Lý Tín Hiệu

## Tổng Quan
Đây là bài tập thực hành thuộc môn **Các phương pháp xử lý tín hiệu**, được thực hiện trong chương trình học tại trường. Bài tập này tập trung vào việc mô phỏng quá trình **tạo ảnh nhòe** (blur) và **khử nhòe** (deblur) sử dụng các phương pháp của bài toán ngược.

## Mục Tiêu
- **Mô phỏng ảnh nhòe** bằng Motion Blur và thêm **nhiễu Gauss** với các mức PSNR (30dB, 20dB, 10dB).
- **Áp dụng các phương pháp khôi phục ảnh**:
  - Lọc Wiener (Wiener Filter).
  - Tikhonov Regularization.
  - Total Variation (TV).
- **Đánh giá chất lượng khôi phục** bằng các chỉ số:
  - SSIM (Structural Similarity Index).
  - Sharpness (Laplacian Variance).

## Cấu Trúc Mã Nguồn
- **Mã chính**: File `main.py` chứa toàn bộ mã nguồn để thực hiện bài tập.
- **Thư viện sử dụng**:
  - `OpenCV (cv2)`: Xử lý ảnh.
  - `NumPy`: Tính toán ma trận.
  - `Matplotlib`: Hiển thị kết quả.
  - `Scikit-image (skimage)`: Các phương pháp khôi phục ảnh và tính toán chỉ số SSIM.

## Cách Chạy Chương Trình

### Yêu cầu:
- **Cài đặt Python 3.x**.
- **Cài đặt các thư viện cần thiết**:
  ```bash
  pip install opencv-python numpy matplotlib scikit-image
  ```
- **Ảnh đầu vào**: Đảm bảo có ảnh đầu vào (ví dụ: `frame_27.jpg`) trong thư mục `/home/trungkien/Documents/Images/`.

### Chạy Chương Trình:
- Mở terminal và chạy:
  ```bash
  python main.py
  ```
Chương trình sẽ:
1. Tạo ảnh nhòe và nhiễu cho từng mức PSNR.
2. Khôi phục ảnh bằng 3 phương pháp (Wiener, Tikhonov, TV).
3. Hiển thị kết quả và lưu vào file (3 file: `result_PSNR_30.png`, `result_PSNR_20.png`, `result_PSNR_10.png`).

## Kết Quả
Đầu ra:
- Mỗi lần lặp (PSNR = 30dB, 20dB, 10dB) sẽ tạo một hình ảnh kết quả bao gồm:
  - Hàng trên: Ảnh gốc và ảnh nhòe + nhiễu.
  - Hàng dưới: 3 ảnh khôi phục (Wiener, Tikhonov, TV).
- Tổng cộng 3 hình ảnh sẽ được lưu:
  - `result_PSNR_30.png`
  - `result_PSNR_20.png`
  - `result_PSNR_10.png`

### Đánh Giá:
- Chỉ số SSIM và Sharpness sẽ được in ra terminal để so sánh hiệu quả của các phương pháp.

## Tác Giả
Bài tập được thực hiện bởi sinh viên trong khuôn khổ môn học **Các phương pháp xử lý tín hiệu**.
