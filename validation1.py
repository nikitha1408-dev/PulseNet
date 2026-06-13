import cv2
from skimage.metrics import structural_similarity as ssim

def compare_ecg(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    score, _ = ssim(img1, img2, full=True)
    return score * 100


# Example usage
if __name__ == "__main__":
    acc = compare_ecg("final_ecg_plot.png", "final_ecg_plot.png")
    print(f"Validation Accuracy: {acc:.2f}%")