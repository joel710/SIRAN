# Siran

Moteur de vision par ordinateur ultra-léger, concu pour la détection NSFW en temps réel et la prévention de contenu explicite non consenti, directement sur l'appareil.

## Apercu

Siran est le sous-système de modération de contenu développé pour le réseau social **Echo**. Concu pour un déploiement haute performance dans des environnements à ressources limitées, il utilise une architecture MobileNetV3 optimisée pour intercepter et analyser les images explicites avant leur ingestion serveur.

Un objectif principal de Siran est l'équité démographique. Par un équilibrage ciblé des datasets et une augmentation agressive de l'espace colorimétrique, le modèle corrige les biais algorithmiques historiques associés aux peaux foncées, garantissant une précision uniforme sur l'ensemble des phénotypes humains.

### Piliers architecturaux

* **Inférence on-device sans latence :** Profil d'exécution quantifié concu pour tourner côté client dans les applications mobiles sans bloquer le thread UI principal.
* **Augmentation éthique des données :** Entrainé pour dissocier la densité de mélanine de la nudité structurelle, éliminant les faux positifs courants dans les modèles open-source standards.
* **Conception privacy-first :** Le traitement du contenu s'effectue localement dans le sandbox de l'application. Aucune donnée image n'est transmise à des API tierces.

---

## Spécifications techniques

Le moteur repose sur un backbone MobileNetV3-Small personnalisé, dont la tête de classification ImageNet standard est remplacée par une topologie de classification binaire spécialisée.

| Métrique | Spécification |
| :--- | :--- |
| **Architecture** | MobileNetV3-Small (modifié) |
| **Dimension du tenseur d'entrée** | 224 x 224 x 3 (RGB) |
| **Taille sur disque (INT8 quantifié)** | ~3.4 Mo |
| **Empreinte mémoire (RAM)** | < 12 Mo en inférence active |
| **Vitesse d'inférence moyenne (CPU)** | ~15ms (ARMv8 milieu de gamme) |
| **Framework cible** | TensorFlow Lite / ONNX Mobile Runtime |

---

## Correction du biais démographique

Les datasets NSFW publics standards présentent un biais significatif vers les distributions démographiques occidentales. Déployés dans des écosystèmes africains, ces modèles génèrent fréquemment des faux positifs par incapacité à distinguer les gradients d'ombre, les environnements à fort contraste, et l'exposition réelle de peau.

SIRAN résout ce problème par une méthodologie d'entrainement en deux volets :

1. **Stratification phénotypique :** Intégration des datasets FairFace et Casual Conversations v2 dans la partition SFW (Safe-For-Work), forcant le réseau à apprendre des représentations faciales et anatomiques diversifiées sous des conditions d'éclairage variées.
2. **Color Jittering & Grayscale stochastique :** Durant la phase d'optimisation, les images subissent des modulations aléatoires de luminosité ($[0.6, 1.4]$), contraste ($[0.6, 1.4]$) et saturation ($[0.8, 1.2]$), ainsi qu'une probabilité de 20% de conversion complète en niveaux de gris. Cela enseigne aux couches convolutives à prioriser la géométrie structurelle et les textures contextuelles plutôt que les valeurs absolues de l'espace colorimétrique.

---

## Structure du dépôt

```text
SIRAN/
├── .github/
│   └── workflows/              # Pipelines CI/CD
├── core/
│   ├── __init__.py
│   ├── model.py                # MobileNetV3-Small avec tête de classification binaire
│   ├── transforms.py           # Augmentations anti-biais (ColorJitter, grayscale)
│   └── train.py                # Boucle d'entrainement (BCEWithLogits, AdamW, CosineAnnealing)
├── dataset/
│   ├── __init__.py
│   └── prepare.py              # Ingestion d'images, split train/val, structuration
├── export/
│   ├── __init__.py
│   └── convert_to_tflite.py    # Conversion PyTorch → ONNX → TFLite quantifié INT8
├── tests/
│   ├── __init__.py
│   ├── test_model.py           # Validation forme et range des sorties du modèle
│   └── test_transforms.py      # Tests de la pipeline d'augmentation
├── .gitignore
├── requirements.txt            # Dépendances Python
├── info.md                     # Stratégie de données et référence anti-biais
└── README.md
```

---

## Démarrage rapide

### Prérequis

* Python 3.10 ou supérieur
* PyTorch 2.0+
* CUDA Toolkit (optionnel, pour l'entrainement accéléré)

### Installation

Cloner le dépôt et installer les dépendances dans un environnement virtuel :

```bash
git clone https://github.com/your-organization/Siran.git
cd Siran
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Préparation du dataset

Structurer les images brutes en split train/val :

```bash
python dataset/prepare.py \
  --input_dir ./data/raw \
  --output_dir ./data/processed \
  --val_ratio 0.15
```

### Entrainement du modèle

Lancer la pipeline d'entrainement avec correction de biais :

```bash
python core/train.py \
  --dataset_path ./data/processed \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.001 \
  --output_dir ./models/checkpoints
```

### Export et quantification

Export ONNX avec quantification INT8 :

```bash
python export/convert_to_tflite.py \
  --checkpoint ./models/checkpoints/best_model.pt \
  --format onnx \
  --quantize \
  --output ./export/siran_v1
```

Conversion TFLite pour déploiement mobile :

```bash
python export/convert_to_tflite.py \
  --checkpoint ./models/checkpoints/best_model.pt \
  --format tflite \
  --output ./export/siran_v1
```

---

## Stratégie de déploiement

Le fichier `.tflite` généré est concu pour être téléchargé dynamiquement par l'application cliente lors de séquences de synchronisation en arrière-plan. Cela minimise la taille initiale du binaire tout en assurant la persistance locale du modèle.

### Matrice de fallback

```
[Image sélectionnée] ──► [SIRAN local ?] ──► OUI ──► Inférence locale (coût zéro)
                                │
                                └──► NON ──► Ingestion API sécurisée ──► Téléchargement async du modèle
```

---

## Licence

Ce projet est distribué sous licence Apache 2.0 - voir le fichier LICENSE pour plus de détails.
