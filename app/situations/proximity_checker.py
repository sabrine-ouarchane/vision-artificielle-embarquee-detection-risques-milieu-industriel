"""
app/situations/proximity_checker.py

Détecte les situations dangereuses de type "proximité homme-machine mobile"
(convoyeur, roue-pelle, etc.), conformément à l'objectif 2.2 du cahier des
charges.

Principe : pour chaque personne détectée, on calcule la distance (en pixels)
jusqu'à la machine la plus proche. Si cette distance passe sous un seuil,
la situation est considérée comme dangereuse.

LIMITE CONNUE (à documenter dans le rapport) : sans calibration caméra
(correspondance pixels <-> mètres réels), la distance est mesurée en pixels,
pas en unité physique. C'est une approximation raisonnable pour un POC, mais
qui dépend de la distance de la caméra à la scène et de la résolution vidéo.
Une calibration ultérieure (connaissant une taille de référence dans le
champ de la caméra) permettrait de convertir en mètres.
"""

import math


def _bbox_edge_distance(bbox_a, bbox_b) -> float:
    """
    Distance la plus courte entre les BORDS de deux boîtes (pas leurs
    centres) — plus représentatif de la vraie proximité physique,
    surtout quand les boîtes ont des tailles très différentes (une
    personne est petite, une machine peut être énorme).

    Retourne 0.0 si les boîtes se chevauchent déjà.
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    dx = max(ax1 - bx2, bx1 - ax2, 0)
    dy = max(ay1 - by2, by1 - ay2, 0)

    return math.hypot(dx, dy)


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(bbox_a, bbox_b) -> float:
    """IoU classique, utilisé pour fusionner les détections 'Machine' en double."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = _bbox_area(bbox_a) + _bbox_area(bbox_b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _clean_machines(machines: list[dict], frame_area: float | None, max_area_ratio: float, iou_threshold: float = 0.3) -> list[dict]:
    """
    Nettoie les détections 'Machine' avant le calcul de proximité :
    1. Filtre les boîtes anormalement grandes (souvent dues au brouillard/
       poussière qui floute les contours réels, faisant "déborder" la boîte
       largement au-delà de l'engin réel) -> évite les faux 0px de distance.
    2. Fusionne les boîtes en double (même engin détecté 2 fois), en ne
       gardant que celle avec la meilleure confiance.
    """
    if frame_area is not None:
        machines = [m for m in machines if _bbox_area(m["bbox"]) <= max_area_ratio * frame_area]

    if len(machines) <= 1:
        return machines

    sorted_machines = sorted(machines, key=lambda m: m["confidence"], reverse=True)
    kept = []
    for machine in sorted_machines:
        if all(_iou(machine["bbox"], k["bbox"]) < iou_threshold for k in kept):
            kept.append(machine)
    return kept


class ProximityChecker:
    """
    Utilisation :
        checker = ProximityChecker(distance_threshold_px=150)
        situations = checker.check(tracked_objects, frame_shape=(height, width))
        # situations = [
        #   {"track_id": 3, "bbox": (...), "nearest_machine_distance": 87.4,
        #    "is_danger": True},
        #   ...
        # ]
    """

    def __init__(self, distance_threshold_px: float = 150, max_machine_area_ratio: float = 0.6):
        """
        Args:
            distance_threshold_px: distance en pixels sous laquelle une
                personne est considérée en situation de proximité dangereuse
                avec une machine. À ajuster selon la résolution de votre
                vidéo et la distance de la caméra (pas de valeur universelle
                sans calibration).
            max_machine_area_ratio: une boîte "Machine" occupant plus de
                cette proportion de la surface de l'image est considérée peu
                fiable (mal délimitée, souvent à cause du brouillard/
                poussière) et ignorée pour le calcul de proximité.
        """
        self.distance_threshold_px = distance_threshold_px
        self.max_machine_area_ratio = max_machine_area_ratio

    def check(self, detections: list[dict], frame_shape: tuple[int, int] | None = None) -> list[dict]:
        """
        Args:
            frame_shape: (height, width) de la frame courante, utilisé pour
                filtrer les boîtes 'Machine' anormalement grandes. Optionnel :
                si omis, aucun filtre de taille n'est appliqué.
        """
        persons = [d for d in detections if d["class_name"] == "Person"]
        raw_machines = [d for d in detections if d["class_name"] == "Machine"]

        frame_area = frame_shape[0] * frame_shape[1] if frame_shape is not None else None
        machines = _clean_machines(raw_machines, frame_area, self.max_machine_area_ratio)

        situations = []
        for person in persons:
            if not machines:
                # Aucune machine dans le champ -> pas de situation de proximité possible
                situations.append({
                    "track_id": person.get("track_id"),
                    "bbox": person["bbox"],
                    "nearest_machine_distance": None,
                    "is_danger": False,
                })
                continue

            distances = [_bbox_edge_distance(person["bbox"], m["bbox"]) for m in machines]
            nearest_distance = min(distances)

            situations.append({
                "track_id": person.get("track_id"),
                "bbox": person["bbox"],
                "nearest_machine_distance": round(nearest_distance, 1),
                "is_danger": nearest_distance < self.distance_threshold_px,
            })

        return situations

    @staticmethod
    def format_danger_message(situation: dict) -> str:
        track_id = situation["track_id"] if situation["track_id"] is not None else "?"
        distance = situation["nearest_machine_distance"]
        return f"Personne #{track_id} : proximité dangereuse avec une machine (distance ≈ {distance}px)"
