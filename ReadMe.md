# YOLOv8 Car Detector GUI

This project is a Python-based graphical interface for detecting **cars** in video using **YOLOv8**.  
It allows users to choose the model, device, and various processing options through an interactive GUI.

---

## 📌 Purpose

This interface was developed for **CTIS testing** purposes by Carlos Ortiz.  
It demonstrates real-time and offline video detection using YOLOv8, with configurable parameters for controlled evaluation.

---

## Features

### Graphical User Interface (GUI)
- Built with Tkinter
- Load video from:
  - Live camera
  - File import dialog
  - Preloaded `/videos` folder

### YOLOv8 Model Selection
Choose from official YOLOv8 models:

| Option Label                   | Model Used      |
|-------------------------------|------------------|
| YOLOv8 Nano                   | `yolov8n.pt`     |
| YOLOv8 Small                  | `yolov8s.pt`     |
| YOLOv8 Medium                 | `yolov8m.pt`     |
| YOLOv8 Large                  | `yolov8l.pt`     |
| YOLOv8 XLarge                 | `yolov8x.pt`     |

### Inference Device Selection
- CPU
- GPU (CUDA), if available

### Optional Settings
- Enable or disable detection logs
- Enable or skip manual ROI cropping:
  - Select 4 points on the first frame of the video to define the region of interest

---

## REQUIREMETNS

## Command-Line Flags

To pre-select a model before launching the GUI, run with one of the following flags:

```bash
python yolo_gui.py --heavy      # Uses yolov8x (Extra Large)
python yolo_gui.py --medium     # Uses yolov8m (Medium)
python yolo_gui.py --light      # Uses yolov8n (Nano)

 

