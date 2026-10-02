"""
extract_frames.py

Extrait une frame toutes les N frames d'une vidéo, et les sauvegarde en
images numérotées. Permet de parcourir visuellement la vidéo dans
l'explorateur de fichiers pour repérer le numéro exact de la frame qui
vous intéresse (ex. celle où une violation semble mal détectée).

Usage :
    python extract_frames.py
"""

import cv2
from pathlib import Path

INPUT_VIDEO = "data/test_videos/homme_machine2.mp4"
OUTPUT_DIR = "data/outputs/frames_preview"
EVERY_N_FRAMES = 10  # extrait 1 frame sur 10 (ajustez si besoin)


def main():
    if not Path(INPUT_VIDEO).exists():
        raise FileNotFoundError(f"Vidéo introuvable : {INPUT_VIDEO}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Vidéo : {total_frames} frames @ {fps:.1f} FPS")
    print(f"[INFO] Extraction d'une frame toutes les {EVERY_N_FRAMES} frames...")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % EVERY_N_FRAMES == 0:
            seconde = frame_idx / fps
            filename = output_dir / f"frame_{frame_idx:05d}_t{seconde:.1f}s.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"\n[OK] {saved_count} images sauvegardées dans : {OUTPUT_DIR}")
    print("Ouvrez ce dossier, repérez la bonne image, et lisez son numéro de frame dans le nom du fichier.")


if __name__ == "__main__":
    main()
