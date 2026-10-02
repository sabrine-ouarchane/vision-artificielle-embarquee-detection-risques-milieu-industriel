"""
test_video.py (v2)

Pipeline complet : détection + tracking + vérification de conformité EPI
+ déclenchement d'alertes HSE en cas de violation (absence de casque/gilet/
harnais, usage du téléphone).

Usage :
    python test_video.py
"""

import cv2
from pathlib import Path
from app.detection.yolo_detector import YoloDetector
from app.tracking.bytetrack_tracker import ByteTrackTracker
from app.tracking.box_smoother import BoxSmoother
from app.epi.compliance_checker import EPIComplianceChecker
from app.alerts.alert_manager import AlertManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_VIDEO = "data/test_videos/harnais_yes.mp4"
OUTPUT_VIDEO = "data/outputs/harnais_yes_annotated.mp4"
ALERTS_LOG = "data/outputs/alerts_log.csv"
MODEL_PATH = "models/best.pt"
CONFIDENCE_THRESHOLD = 0.5
ALERT_COOLDOWN_FRAMES = 90  # ~3s à 30 FPS, évite le spam d'alertes

# Mettez True si test1.mp4 montre une scène de travail en hauteur
# (le harnais devient alors obligatoire, sinon il est juste suivi/affiché)
HARNESS_REQUIRED = True

# Lissage temporel des boîtes : plus bas = plus stable mais plus lent à réagir
SMOOTHING_ALPHA = 0.5

COLOR_COMPLIANT = (0, 200, 0)       # vert : personne en conformité (EPI positivement détecté)
COLOR_VIOLATION = (0, 0, 255)       # rouge : violation détectée
COLOR_UNDETERMINED = (0, 165, 255)  # orange : aucune preuve suffisante (ni EPI, ni absence détectés)


def draw_person_status(frame, status: dict):
    """Dessine la boîte d'une personne selon son statut : vert/rouge/orange."""
    x1, y1, x2, y2 = status["bbox"]
    track_id = status["track_id"] if status["track_id"] is not None else "?"

    if status["status"] == "violation":
        color = COLOR_VIOLATION
        readable = ", ".join(status["violations"])
        label = f"Personne #{track_id} - VIOLATION ({readable})"
    elif status["status"] == "compliant":
        color = COLOR_COMPLIANT
        label = f"Personne #{track_id} - OK"
    else:  # undetermined
        color = COLOR_UNDETERMINED
        label = f"Personne #{track_id} - INDETERMINE"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    if not Path(INPUT_VIDEO).exists():
        raise FileNotFoundError(f"Vidéo introuvable : {INPUT_VIDEO}")
    Path("data/outputs").mkdir(parents=True, exist_ok=True)

    tracker = ByteTrackTracker(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    smoother = BoxSmoother(alpha=SMOOTHING_ALPHA, max_missed_frames=15)
    checker = EPIComplianceChecker(harness_required=HARNESS_REQUIRED)
    alert_manager = AlertManager(log_path=ALERTS_LOG, cooldown_frames=ALERT_COOLDOWN_FRAMES)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Vidéo : {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    # Codec plus largement compatible avec les lecteurs Windows par défaut
    # (mp4v pose parfois problème avec le Lecteur Windows Media / app Photos)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[ATTENTION] Le codec 'avc1' n'est pas disponible sur ce système, "
              "retour au codec 'mp4v' (peut nécessiter VLC pour la lecture).")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_idx = 0
    total_alerts = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        tracked_objects = tracker.track(frame, persist=True)
        tracked_objects = smoother.smooth(tracked_objects, frame_idx)
        statuses = checker.check(tracked_objects)

        for status in statuses:
            draw_person_status(frame, status)

            if not status["is_compliant"]:
                message = checker.format_violation_message(status)
                sent = alert_manager.send_alert(
                    track_id=status["track_id"],
                    message=message,
                    frame_idx=frame_idx,
                )
                if sent:
                    total_alerts += 1

        writer.write(frame)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"[INFO] Frame {frame_idx}/{total_frames} traitée ({len(statuses)} personne(s) suivie(s))")

    cap.release()
    writer.release()

    print(f"\n[OK] Vidéo annotée sauvegardée : {OUTPUT_VIDEO}")
    print(f"[OK] {total_alerts} alerte(s) déclenchée(s), journal complet dans : {ALERTS_LOG}")


if __name__ == "__main__":
    main()