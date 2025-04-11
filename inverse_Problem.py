import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim

# =========================
# Bộ hàm xử lý hình ảnh
# =========================

def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy

def motion_blur_kernel(size=15, angle=0):
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            dist = abs((i - center) * np.cos(np.radians(angle)) +
                       (j - center) * np.sin(np.radians(angle)))
            if dist < 0.5:
                kernel[i, j] = 1
    return kernel / np.sum(kernel)

def apply_motion_blur(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def tikhonov_deconvolution(img, kernel, lamb=0.01):
    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel, s=img.shape)
    H_conj = np.conj(kernel_fft)
    denominator = np.abs(kernel_fft) ** 2 + lamb
    result_fft = H_conj * img_fft / denominator
    result = np.fft.ifft2(result_fft).real
    return np.clip(result, 0, 1)

def compute_ssim(ref, test):
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    return ssim(ref_gray, test_gray, data_range=255)

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return np.var(lap)

# =========================
# Xử lý chính
# =========================

def restore_image_channels(blurred, kernel, lamb=0.01, tv_weight=0.05):
    methods = {'wiener': [], 'tikhonov': [], 'tv': []}
    for ch in range(3):
        ch_blurred = blurred[:, :, ch] / 255.0
        methods['wiener'].append(
            restoration.wiener(ch_blurred, kernel, balance=0.5))
        methods['tikhonov'].append(
            tikhonov_deconvolution(ch_blurred, kernel, lamb))
        methods['tv'].append(
            denoise_tv_chambolle(ch_blurred, weight=tv_weight))
    return {
        key: (np.stack(val, axis=-1) * 255).astype(np.uint8)
        for key, val in methods.items()
    }

def process_restoration(original, noisy_blur, kernel):
    restored = restore_image_channels(noisy_blur, kernel)
    metrics = {}
    for method, img in restored.items():
        metrics[method] = {
            'image': img,
            'ssim': compute_ssim(original, img),
            'sharpness': compute_sharpness(img)
        }
    return metrics

def show_and_save_results(original, blurred, metrics, title_prefix="", output_path="result.png"):
    # Tạo figure với bố cục 2x3
    fig = plt.figure(figsize=(15, 8))
    
    # Hàng trên: Ảnh gốc và ảnh nhòe
    plt.subplot(2, 3, 1)
    plt.title("Original Image", fontsize=25)  # Tăng kích thước font
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Blurred + Noisy", fontsize=25
    )  # Tăng kích thước font
    plt.imshow(blurred)
    plt.axis("off")

    # Hàng dưới: 3 ảnh khôi phục
    method_names = ['wiener', 'tikhonov', 'tv']
    for i, method in enumerate(method_names, start=4):
        result = metrics[method]
        plt.subplot(2, 3, i)
        plt.title(f"{method.capitalize()}\nSSIM={result['ssim']:.4f} | Sharpness={result['sharpness']:.2f}", fontsize=20)  # Tăng kích thước font
        plt.imshow(result['image'])
        plt.axis("off")
    
    plt.suptitle(title_prefix, fontsize=35)  # Tăng kích thước font cho tiêu đề
    plt.tight_layout()

    # Lưu hình ảnh vào file
    fig.canvas.draw()
    # Chuyển đổi figure thành numpy array
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Lưu bằng cv2.imwrite
    cv2.imwrite(output_path, cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR))
    print(f"Đã lưu kết quả vào: {output_path}")

    # Hiển thị kết quả
    plt.show()

# =========================
# Thực thi chương trình
# =========================

def main():
    image_path = '/home/trungkien/Documents/Images/frame_27.jpg'
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Không thể mở ảnh.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    psnr_levels = {
        30: 0.01,
        20: 25,
        10: 75
    }

    kernel = motion_blur_kernel(size=30, angle=0)

    for psnr, noise_sigma in psnr_levels.items():
        noisy_image = add_gaussian_noise(image, sigma=noise_sigma)
        blurred_noisy = apply_motion_blur(noisy_image, kernel)
        metrics = process_restoration(image, blurred_noisy, kernel)
        print(f"\n--- PSNR = {psnr} dB ---")
        for method in metrics:
            print(f"{method.capitalize()}: SSIM = {metrics[method]['ssim']:.4f}, Sharpness = {metrics[method]['sharpness']:.2f}")

        # Lưu kết quả vào file (bao gồm cả 5 phần: ảnh gốc, ảnh nhòe, và 3 ảnh khôi phục)
        output_path = f"result_PSNR_{psnr}.png"
        show_and_save_results(image, blurred_noisy, metrics, 
                              title_prefix=f"Ảnh được thêm nhiễu Gauss với PSNR {psnr} dB",
                              output_path=output_path)

if __name__ == "__main__":
    main()
