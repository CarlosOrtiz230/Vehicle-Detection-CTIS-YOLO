# yolo_gui.py (Updated & Fixed)
import cv2
import os
import logging
import argparse
import threading
import numpy as np
from tkinter import Tk, Label, Button, filedialog, StringVar, OptionMenu, Listbox, SINGLE, END
from ultralytics import YOLO
from pathlib import Path
from sort import Sort
from datetime import datetime

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

trackers = ["None", "SORT", "DeepSORT"]

# ========== Crop & Line Selection ==========
def select_points_on_image(image, prompt="Select Points", num_points=4):
    clone = image.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(prompt, clone)

    cv2.imshow(prompt, clone)
    cv2.setMouseCallback(prompt, click_event)

    while len(points) < num_points:
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return points

def warp_perspective(frame, crop_pts):
    dst_size = (500, 500)
    try:
        pts_src = np.array(crop_pts, dtype="float32")
        pts_dst = np.array([
            [0, 0], [dst_size[0] - 1, 0],
            [dst_size[0] - 1, dst_size[1] - 1], [0, dst_size[1] - 1]
        ], dtype="float32")
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(frame, matrix, dst_size)
        return warped, matrix
    except Exception as e:
        print("warpPerspective failed:", e)
        return frame, None

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def line_crossed(p1, p2, line):
    def side(p, a, b):
        return np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
    return side(p1, *line) != side(p2, *line)

# ========== GUI ==========
class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 CTIS Car Detector")
        self.root.geometry("650x560")

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

        Label(root, text="Tracking Mode:").pack()
        self.tracker_choice = StringVar(value="SORT")
        OptionMenu(root, self.tracker_choice, *trackers).pack(pady=5)

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
        return YOLO(model_file).to(device)

    def run_yolo(self, source):
        model = self.get_model()
        tracker = Sort() if self.tracker_choice.get() == "SORT" else None

        cap = cv2.VideoCapture(0 if source == "camera" else source)
        ret, frame = cap.read()
        if not ret:
            print("Error reading video.")
            return

        crop_pts, line_pts, matrix = None, None, None
        if source != "camera" and self.crop_enabled.get() == "Enable Crop":
            crop_pts = select_points_on_image(frame.copy(), "Select 4 Crop Points", num_points=4)
            if len(crop_pts) == 4:
                preview_crop, matrix = warp_perspective(frame.copy(), crop_pts)
                cv2.imshow("Cropped Preview", preview_crop)
                cv2.waitKey(1000)
                cv2.destroyWindow("Cropped Preview")
                line_pts = select_points_on_image(preview_crop.copy(), "Draw Line (2 Points)", num_points=2)
                print(f"Line Points: {line_pts}")


        used_ids, id_positions, count = set(), {}, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if matrix is not None:
                frame = cv2.warpPerspective(frame, matrix, (500, 500))

            result = model(frame, verbose=False)[0]
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if model.names[cls_id] == "car":
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf.item())
                    detections.append([x1, y1, x2, y2, conf])

            if tracker and detections:
                tracks = tracker.update(np.array(detections))
                for x1, y1, x2, y2, track_id in tracks:
                    cx, cy = get_center([x1, y1, x2, y2])
                    if track_id in id_positions:
                        prev = id_positions[track_id]
                        if line_pts and line_crossed(prev, (cx, cy), line_pts) and track_id not in used_ids:
                            count += 1
                            used_ids.add(track_id)
                            with open("counter.log", "a") as f:
                                f.write(f"{datetime.now()} - ID {int(track_id)} crossed\n")
                    id_positions[track_id] = (cx, cy)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f'car ID {int(track_id)}'    
                    cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                for det in detections:
                    x1, y1, x2, y2, _ = map(int, det[:5])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)

            cv2.putText(frame, f"Count: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if line_pts:
                cv2.line(frame, line_pts[0], line_pts[1], (0, 0, 255), 2)

            cv2.imshow("YOLOv8 Detection", frame)
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

# ========== Logging ==========
logging.basicConfig(
    filename='detections.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# ========== Launch ==========
if __name__ == "__main__":
    root = Tk()
    app = YOLOApp(root)
    root.mainloop()
