"""
app/epi/compliance_checker.py

Associe chaque personne détectée (Person) à ses équipements de protection
individuelle (Hardhat/NO-Hardhat, Safety Vest/NO-Safety Vest, Harness/NO-Harness)
et à un éventuel usage de téléphone, afin de déterminer un statut de conformité.

Principe d'association :
    YOLOv8 détecte chaque objet indépendamment (une boîte "Person", une boîte
    "NO-Hardhat", etc.). Ce module rattache chaque EPI/violation à la bonne
    personne en se basant sur le chevauchement spatial des boîtes (la boîte de
    l'EPI doit être largement contenue dans la boîte de la personne).
"""

# Classes qui indiquent une violation OBLIGATOIRE pour toute personne détectée.
# Le harnais N'EST PAS dans cette liste : il n'est requis que pour le travail
# en hauteur (>1,80m, cf. standard S-HSE-SEC-29), donc son absence ne sera
# comptée comme violation que plus tard, une fois croisée avec le zonage
# (zone "travail en hauteur"). Pour l'instant NO-Harness est ignoré ici.
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest", "Phone"}

# Classes suivies mais non comptées comme violation obligatoire pour l'instant
# (utilisées plus tard avec la logique de zonage)
CONDITIONAL_VIOLATION_CLASSES = {"NO-Harness"}

# Classes qui indiquent une conformité (EPI bien porté)
COMPLIANT_CLASSES = {"Hardhat", "Safety Vest", "Harness"}

# EPI dont la présence DOIT être confirmée simultanément (ET, pas OU) pour
# qu'une personne soit déclarée "conforme". Le harnais n'est pas dans cette
# liste de base : il ne devient requis (via harness_required) que pour les
# zones de travail en hauteur.
REQUIRED_FOR_COMPLIANT = {"Hardhat", "Safety Vest"}

# Association classe -> équipement concerné (pour affichage lisible)
EQUIPMENT_LABELS = {
    "Hardhat": "casque",
    "NO-Hardhat": "casque",
    "Safety Vest": "gilet",
    "NO-Safety Vest": "gilet",
    "Harness": "harnais",
    "NO-Harness": "harnais",
    "Phone": "téléphone",
}


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _overlap_ratio(inner_bbox, outer_bbox):
    """
    Calcule quelle proportion de inner_bbox est contenue dans outer_bbox.
    Utilisé pour savoir à quelle personne appartient réellement un EPI,
    même quand plusieurs personnes sont proches/superposées à l'écran.
    """
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    inner_area = _bbox_area(inner_bbox)

    if inner_area == 0:
        return 0.0
    return inter_area / inner_area


def _iou(bbox_a, bbox_b):
    """
    Intersection over Union classique (symétrique), plus fiable que
    _overlap_ratio pour détecter des doublons de personnes quand les deux
    boîtes ont des tailles très différentes (ex: une boîte couvre juste le
    haut du corps, l'autre couvre le corps entier).
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def _deduplicate_persons(persons: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """
    Fusionne les détections "Person" en double : quand deux boîtes se
    chevauchent fortement (même individu détecté deux fois, souvent à cause
    d'un changement de pose ou d'un track_id qui bascule), ne garde que
    celle avec la meilleure confiance.

    Utilise à la fois l'IoU classique ET le ratio de containment (le plus
    élevé des deux), pour couvrir aussi bien les doublons "même taille
    décalée" que les doublons "une boîte contient l'autre".
    """
    if len(persons) <= 1:
        return persons

    sorted_persons = sorted(persons, key=lambda p: p["confidence"], reverse=True)
    kept = []

    for person in sorted_persons:
        is_duplicate = False
        for kept_person in kept:
            iou_score = _iou(person["bbox"], kept_person["bbox"])
            containment_score = _overlap_ratio(person["bbox"], kept_person["bbox"])
            if max(iou_score, containment_score) >= iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(person)

    return kept


class EPIComplianceChecker:
    """
    Prend en entrée les détections brutes d'une frame (issues de YoloDetector
    ou ByteTrackTracker) et produit un statut de conformité par personne.

    Utilisation :
        checker = EPIComplianceChecker()
        statuses = checker.check(tracked_objects)
        # statuses = [
        #   {"track_id": 3, "bbox": (...), "violations": ["NO-Hardhat", "Phone"],
        #    "compliant_items": ["Safety Vest"], "is_compliant": False},
        #   ...
        # ]
    """

    def __init__(self, harness_required: bool = False):
        """
        Args:
            harness_required: True si le contexte analysé est une zone de
                travail en hauteur (>1,80m, cf. S-HSE-SEC-29) où le harnais
                devient obligatoire. Dans ce cas, NO-Harness est traité comme
                une violation au même titre que NO-Hardhat/NO-Safety Vest/Phone.
                False par défaut (zone au sol classique, harnais non exigé).

                Pour l'instant ce réglage est manuel, par vidéo/scène. Une fois
                le zonage automatique construit (Étape 2.3), cette valeur
                pourra être déterminée dynamiquement selon la zone où se
                trouve la personne, plutôt que fixée pour toute la vidéo.
        """
        self.harness_required = harness_required

    # Chevauchement minimal requis pour attribuer un EPI à une personne
    MIN_OVERLAP_RATIO = 0.5

    def check(self, detections: list[dict]) -> list[dict]:
        raw_persons = [d for d in detections if d["class_name"] == "Person"]
        persons = _deduplicate_persons(raw_persons)  # fusionne les doublons (même personne, 2 boîtes)
        equipment_items = [d for d in detections if d["class_name"] != "Person"]

        # Étape 1 : pour chaque EPI, trouver LA personne avec le meilleur
        # chevauchement (et une seule), pour éviter qu'un même EPI soit
        # attribué à plusieurs personnes proches/superposées à l'écran.
        assignments: dict[int, list[dict]] = {i: [] for i in range(len(persons))}

        for item in equipment_items:
            best_person_idx = None
            best_ratio = 0.0
            for idx, person in enumerate(persons):
                ratio = _overlap_ratio(item["bbox"], person["bbox"])
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_person_idx = idx

            if best_person_idx is not None and best_ratio >= self.MIN_OVERLAP_RATIO:
                assignments[best_person_idx].append(item)

        # Étape 2 : construire le statut de conformité de chaque personne
        # à partir des EPI qui lui ont été assignés.
        statuses = []
        for idx, person in enumerate(persons):
            violations = []
            conditional_violations = []
            compliant_items = []

            for item in assignments[idx]:
                class_name = item["class_name"]
                if class_name in VIOLATION_CLASSES:
                    violations.append(class_name)
                elif class_name in CONDITIONAL_VIOLATION_CLASSES:
                    if self.harness_required:
                        violations.append(class_name)
                    else:
                        conditional_violations.append(class_name)
                elif class_name in COMPLIANT_CLASSES:
                    compliant_items.append(class_name)

            # Détermination du statut à 3 états :
            # - "violation" : OU logique -> UNE SEULE absence détectée suffit
            #   (principe de précaution HSE : mieux vaut alerter à tort que
            #   rater une vraie violation)
            # - "compliant" : ET logique -> TOUS les EPI requis (casque ET
            #   gilet) doivent être positivement confirmés, pas un seul
            # - "undetermined" : ni violation, ni confirmation complète
            #   (preuve partielle ou absente -> on ne doit pas afficher "OK")
            compliant_classes_found = set(compliant_items)

            # Liste des EPI requis pour "CONFORME" : casque + gilet toujours,
            # + harnais en plus SI on est en contexte travail en hauteur
            # (cohérent avec le fait que NO-Harness devient aussi une
            # violation dans ce même contexte, cf. harness_required)
            required_now = set(REQUIRED_FOR_COMPLIANT)
            if self.harness_required:
                required_now.add("Harness")

            all_required_confirmed = required_now.issubset(compliant_classes_found)

            if len(violations) > 0:
                status_label = "violation"
            elif all_required_confirmed:
                status_label = "compliant"
            else:
                status_label = "undetermined"

            statuses.append({
                "track_id": person.get("track_id"),
                "bbox": person["bbox"],
                "confidence": person["confidence"],
                "violations": violations,
                "conditional_violations": conditional_violations,
                "compliant_items": compliant_items,
                "is_compliant": len(violations) == 0,
                "status": status_label,  # "violation" | "compliant" | "undetermined"
            })

        return statuses

    @staticmethod
    def format_violation_message(status: dict) -> str:
        """Construit un message lisible pour une alerte."""
        readable = [EQUIPMENT_LABELS.get(v, v) for v in status["violations"]]
        track_id = status["track_id"] if status["track_id"] is not None else "?"
        return f"Personne #{track_id} : violation détectée ({', '.join(readable)})"