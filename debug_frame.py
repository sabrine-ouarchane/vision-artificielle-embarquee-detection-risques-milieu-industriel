"""
debug_frame.py

Extrait UNE frame d'une vidéo et affiche les détections BRUTES du modèle
(sans tracking, sans logique de conformité), avec la confiance de chaque
boîte. Utile pour diagnostiquer si un problème vient du modèle lui-même
(boîte mal délimitée, classe mal reconnue) ou de la logique appliquée
ensuite (association EPI, tracking).

Usage :
    python debug_frame.py
"""

import cv2
from pathlib import Path
from app.detection.yolo_detector import YoloDetector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_VIDEO = "data/test_videos/homme_machine.mp4"
FRAME_NUMBER = 50   # numéro de la frame à inspecter (0 = première)
OUTPUT_IMAGE = "data/outputs/debug_frame.jpg"
MODEL_PATH = "models/best.pt"
CONFIDENCE_THRESHOLD = 0.3  # volontairement bas pour tout voir, y compris le bruit
IMGSZ = 960

CLASS_COLORS = {
    "Person": (255, 200, 0),
    "Hardhat": (0, 255, 0),
    "NO-Hardhat": (0, 0, 255),
    "Safety Vest": (0, 200, 255),
    "NO-Safety Vest": (0, 0, 255),
    "Harness": (255, 0, 255),
    "NO-Harness": (0, 0, 255),
    "Phone": (0, 255, 255),
    "Machine": (0, 128, 0),
}
DEFAULT_COLOR = (200, 200, 200)


def main():
    if not Path(INPUT_VIDEO).exists():
        raise FileNotFoundError(f"Vidéo introuvable : {INPUT_VIDEO}")
    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Impossible de lire la frame {FRAME_NUMBER}")

    detector = YoloDetector(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    detections = detector.detect(frame)

    print(f"\n[DEBUG] {len(detections)} détection(s) brute(s) sur la frame {FRAME_NUMBER} :\n")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']:<18} conf={det['confidence']:.2f}  bbox={det['bbox']}")

        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(det["class_name"], DEFAULT_COLOR)
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(OUTPUT_IMAGE, frame)
    print(f"\n[OK] Image annotée sauvegardée : {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
