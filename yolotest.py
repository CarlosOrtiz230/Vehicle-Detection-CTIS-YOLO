import cv2
import os
import logging
import argparse
import threading
import numpy as np
from tkinter import Tk, Label, Button, filedialog, StringVar, OptionMenu, Listbox, SINGLE, END
from ultralytics import YOLO
from pathlib import Path

# ========== Command-Line Flags ==========
parser = argparse.ArgumentParser(description="YOLOv8 GUI Video Detection")
parser.add_argument("--heavy", action="store_true", help="Use YOLOv8x (extra large)")
parser.add_argument("--medium", action="store_true", help="Use YOLOv8m (medium)")
parser.add_argument("--light", action="store_true", help="Use YOLOv8n (nano/light)")
args = parser.parse_args()

# ========== Setup ==========
videos_folder = os.path.join(os.getcwd(), 'videos')
os.makedirs(videos_folder, exist_ok=True)

models = {
    "YOLOv8 Nano (yolov8n.pt)": "yolov8n.pt",
    "YOLOv8 Small (yolov8s.pt)": "yolov8s.pt",
    "YOLOv8 Medium (yolov8m.pt)": "yolov8m.pt",
    "YOLOv8 Large (yolov8l.pt)": "yolov8l.pt",
    "YOLOv8 XLarge (yolov8x.pt)": "yolov8x.pt"
}

devices = {
    "CPU": "cpu",
    "GPU (CUDA)": "cuda"
}

# ========== Crop Helpers ==========
def select_crop_points(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Could not read first frame.")
        return None

    clone = frame.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 Points (TL, TR, BR, BL)", clone)

    cv2.imshow("Select 4 Points (TL, TR, BR, BL)", clone)
    cv2.setMouseCallback("Select 4 Points (TL, TR, BR, BL)", click_event)

    while len(points) < 4:
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return points

def warp_perspective(frame, points):
    dst_size = (500, 500)
    pts_src = np.array(points, dtype="float32")
    pts_dst = np.array([
        [0, 0],
        [dst_size[0] - 1, 0],
        [dst_size[0] - 1, dst_size[1] - 1],
        [0, dst_size[1] - 1]
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(frame, matrix, dst_size)

# ========== GUI ==========
class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Car Detector")
        self.root.geometry("620x500")

        # Model override by flag
        if args.heavy:
            self.default_model = "YOLOv8 XLarge (yolov8x.pt)"
        elif args.medium:
            self.default_model = "YOLOv8 Medium (yolov8m.pt)"
        elif args.light:
            self.default_model = "YOLOv8 Nano (yolov8n.pt)"
        else:
            self.default_model = list(models.keys())[0]

        Label(root, text="Select YOLOv8 Model:").pack()
        self.model_choice = StringVar(value=self.default_model)
        OptionMenu(root, self.model_choice, *models.keys()).pack(pady=5)

        Label(root, text="Select Device:").pack()
        self.device_choice = StringVar(value="CPU")
        OptionMenu(root, self.device_choice, *devices.keys()).pack(pady=5)

        self.enable_logging = StringVar(value="Enable Logs")
        OptionMenu(root, self.enable_logging, "Enable Logs", "Disable Logs").pack(pady=5)

        self.crop_enabled = StringVar(value="Enable Crop")
        OptionMenu(root, self.crop_enabled, "Enable Crop", "Disable Crop").pack(pady=5)

        Label(root, text="Choose video source:").pack(pady=10)
        Button(root, text="Use Camera", command=self.use_camera).pack(pady=5)
        Button(root, text="Import Video File", command=self.import_video).pack(pady=5)

        Label(root, text="Or select from /videos folder:").pack(pady=10)
        self.video_list = Listbox(root, selectmode=SINGLE, width=50)
        self.video_list.pack()
        self.refresh_video_list()

        Button(root, text="Run Detection", command=self.run_selected_video).pack(pady=20)

    def refresh_video_list(self):
        self.video_list.delete(0, END)
        for f in os.listdir(videos_folder):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                self.video_list.insert(END, f)

    def get_model(self):
        selected_model = self.model_choice.get()
        model_file = models[selected_model]
        device = devices[self.device_choice.get()]
        cache_path = Path.home() / ".cache" / "ultralytics" / model_file
        if cache_path.exists():
            print(f"✔ Model cached: {model_file}")
        else:
            print(f"⬇ Downloading model: {model_file}")
        return YOLO(model_file).to(device)

    def run_yolo(self, source):
        model = self.get_model()
        crop_points = None
        if source != "camera" and self.crop_enabled.get() == "Enable Crop":
            crop_points = select_crop_points(source)

        cap = cv2.VideoCapture(0 if source == "camera" else source)
        if not cap.isOpened():
            print("Error opening video.")
            return

        print("🟢 Running detection...")
        class_names = model.names

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if crop_points and len(crop_points) == 4:
                frame = warp_perspective(frame, crop_points)

            results = model(frame)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]
                if cls_name == 'car':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{cls_name}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if self.enable_logging.get() == "Enable Logs":
                        logging.info(f'Detection: {cls_name} at ({x1}, {y1}, {x2}, {y2})')

            cv2.imshow(f'YOLOv8 - {Path(source).name if source != "camera" else "Camera"}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def use_camera(self):
        threading.Thread(target=self.run_yolo, args=("camera",), daemon=True).start()

    def import_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            threading.Thread(target=self.run_yolo, args=(file_path,), daemon=True).start()

    def run_selected_video(self):
        selected = self.video_list.curselection()
        if selected:
            filename = self.video_list.get(selected[0])
            full_path = os.path.join(videos_folder, filename)
            threading.Thread(target=self.run_yolo, args=(full_path,), daemon=True).start()

# ========== Logging Init (after GUI to obey flag) ==========
logging.basicConfig(
    filename='detections.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
)

# ========== Run GUI ==========
if __name__ == "__main__":
    root = Tk()
    app = YOLOApp(root)
    root.mainloop()
