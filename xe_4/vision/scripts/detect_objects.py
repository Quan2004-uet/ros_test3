from ultralytics import YOLO
import cv2
import os

# Load model đã train
model_path = os.path.expanduser("~/test_ws/src/xe_4/vision/models/best.pt")
model = YOLO(model_path)

# Đường dẫn ảnh cần nhận diện
image_dir = os.path.expanduser("~/test_ws/src/xe_4/vision/dataset/images/train")

# Lưu ảnh đầu ra
output_dir = os.path.join(image_dir, "predicted")
os.makedirs(output_dir, exist_ok=True)

# Lặp qua các ảnh trong thư mục
for img_file in os.listdir(image_dir):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        img_path = os.path.join(image_dir, img_file)

        # Nhận diện
        results = model(img_path)

        # Hiển thị và lưu kết quả
        annotated = results[0].plot()
        cv2.imshow("Prediction", annotated)
        cv2.imwrite(os.path.join(output_dir, img_file), annotated)
        print(f"Saved result: {os.path.join(output_dir, img_file)}")

        # Thoát nếu nhấn Esc
        if cv2.waitKey(0) & 0xFF == 27:
            break

cv2.destroyAllWindows()
