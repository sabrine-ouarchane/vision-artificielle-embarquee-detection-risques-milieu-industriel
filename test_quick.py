from app.detection.yolo_detector import YoloDetector

detector = YoloDetector("models/best.pt")
print("Classes du modèle :", detector.get_class_names())
