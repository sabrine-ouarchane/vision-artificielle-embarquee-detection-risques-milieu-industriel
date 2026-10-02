# Vision Artificielle Embarquée pour la Détection Proactive des Comportements, Situations et Zones à Risque en Milieu Industriel

Preuve de concept (POC) de surveillance intelligente basée sur **YOLOv8**. Le système analyse des flux vidéo en temps réel pour détecter les personnes, les EPI (casques, gilets, harnais), les téléphones et les engins, suit les objets d'une image à l'autre, applique un zonage dynamique du risque et génère des alertes pour les responsables HSE.

> Projet réalisé lors d'un stage de fin d'études d'ingénieur en IA et cybersécurité (ENSA Béni Mellal), année 2025/2026.

## Contexte et problématique

Sur les sites industriels, malgré des procédures HSE strictes, des risques persistent : non-port des EPI, usage du téléphone en zone de production, intrusions, proximité homme-machine. La surveillance traditionnelle (rondes, vidéosurveillance classique) dépend de l'humain et a ses limites : fatigue, multiplicité des caméras, détection tardive.

L'objectif de ce projet est de passer d'une sécurité **réactive** à une approche **prédictive et proactive**, grâce à l'analyse automatique des flux vidéo par vision par ordinateur.

## Fonctionnalités

- **Détection d'objets** avec YOLOv8 : personnes, casques, gilets, harnais, téléphones, engins
- **Suivi (tracking)** avec ByteTrack et lissage des boîtes de détection
- **Vérification du port des EPI** (module de conformité)
- **Zonage dynamique du risque** : zones définies dans `config/zones.json` et détection des intrusions
- **Situations dangereuses** : proximité homme-machine
- **Capteurs simulés** : capteur de gaz et état des machines
- **Alertes en temps réel** pour les responsables HSE

## Méthodologie

1. Analyse des risques et définition des besoins
2. Collecte et annotation des données
3. Entraînement du modèle YOLOv8 (comparaison de YOLOv8n et YOLOv8s)
4. Développement du prototype (tracking et zonage)
5. Évaluation des performances

## Résultats

Résultats du modèle **YOLOv8s**, entraîné sur 136 époques avant déclenchement de l'EarlyStopping :

| Métrique | Valeur approximative |
|---|---|
| Précision | 0,80 à 0,85 |
| Rappel | 0,63 à 0,65 |
| mAP50 | ≈ 0,74 |
| mAP50-95 | ≈ 0,47 |

Les pertes d'entraînement et de validation (box_loss, cls_loss, dfl_loss) décroissent régulièrement, ce qui montre une bonne convergence. YOLOv8s donne des performances légèrement supérieures à YOLOv8n, au prix d'un temps d'entraînement et d'un coût de calcul plus élevés : ce compromis a guidé le choix du modèle final.

*À COMPLÉTER : le modèle finalement retenu, les résultats de YOLOv8n pour comparaison, le nombre d'images du jeu de données et les classes détectées.*

*À COMPLÉTER : ajoute les courbes d'entraînement et 2 ou 3 captures de la détection en action, dans un dossier `images/` :*
`![Courbes d'entraînement](images/courbes_yolov8s.png)`

## Architecture du projet

```
├── app/
│   ├── detection/    # détection YOLO
│   ├── tracking/     # suivi ByteTrack et lissage des boîtes
│   ├── epi/          # vérification de la conformité des EPI
│   ├── zones/        # contrôle des zones à risque
│   ├── situations/   # situations dangereuses (proximité)
│   ├── sensors/      # capteurs simulés (gaz, état des machines)
│   └── alerts/       # gestion des alertes
├── config/
│   └── zones.json    # définition des zones à risque
├── models/           # modèle(s) de détection
├── dashboard/        # tableau de bord
├── test_*.py         # scripts de test sur vidéo
└── extract_frames.py # extraction d'images depuis une vidéo
```

## Technologies

Python, YOLOv8 (Ultralytics), ByteTrack, OpenCV.



## Installation

```bash
git clone https://github.com/sabrine-ouarchane/vision-artificielle-embarquee-detection-risques-milieu-industriel.git
cd vision-artificielle-embarquee-detection-risques-milieu-industriel
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

*À COMPLÉTER : ajoute un fichier `requirements.txt` à la racine (commande : `pip freeze > requirements.txt`).*

## Utilisation

```bash
python test_video.py
```


## Données

Le dossier `data/` n'est pas inclus dans le dépôt à cause de sa taille (environ 1,7 Go) et de la nature des vidéos.



## Limites et perspectives

Le projet est une preuve de concept. Des pistes d'industrialisation : remplacer les capteurs simulés par de vrais capteurs, déployer sur une carte embarquée, et intégrer un tableau de bord en temps réel.



## Auteure

**Sabrine Ouarchane**, élève ingénieure en IA et cybersécurité, ENSA Béni Mellal.
