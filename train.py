"""
Fine-tune YOLOv8 on your labeled rider-safety / package-condition dataset.
Runs fine on a free Google Colab GPU runtime.

Usage:
    python train.py --data dataset.yaml --epochs 50 --imgsz 640
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base", default="yolov8n.pt", help="pretrained base weights")
    args = parser.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        # basic augmentation to help with low light / occlusion edge cases
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
    )

    metrics = model.val()
    print("Validation metrics:", metrics.results_dict)


if __name__ == "__main__":
    main()
