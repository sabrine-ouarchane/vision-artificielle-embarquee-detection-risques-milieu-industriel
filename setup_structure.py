"""
Génère l'arborescence complète du projet hse-vision-poc.
A exécuter une seule fois, depuis la racine du projet (hse-vision-poc/).
"""
import os

# Dossiers à créer
FOLDERS = [
    "app/detection",
    "app/tracking",
    "app/epi",
    "app/zones",
    "app/sensors",
    "app/alerts",
    "app/database",
    "app/api",
    "config",
    "models",
    "dashboard",
    "data/test_videos",
    "data/outputs",
]

# Fichiers __init__.py pour que les dossiers app/* soient des packages Python
INIT_FILES = [
    "app/__init__.py",
    "app/detection/__init__.py",
    "app/tracking/__init__.py",
    "app/epi/__init__.py",
    "app/zones/__init__.py",
    "app/sensors/__init__.py",
    "app/alerts/__init__.py",
    "app/database/__init__.py",
    "app/api/__init__.py",
]

for folder in FOLDERS:
    os.makedirs(folder, exist_ok=True)
    print(f"[OK] Dossier créé : {folder}")

for init_file in INIT_FILES:
    if not os.path.exists(init_file):
        with open(init_file, "w", encoding="utf-8") as f:
            f.write("")
        print(f"[OK] Fichier créé : {init_file}")

print("\nArborescence prête. N'oubliez pas de placer votre best.pt dans models/")
