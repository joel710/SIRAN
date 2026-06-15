# SIRAN - Stratégie de données d'entrainement et correction de biais

## Objectif

Corriger le biais démographique du classifieur MobileNetV3 pour garantir une précision de détection uniforme sur toutes les carnations, avec un accent particulier sur les populations à peau foncée représentatives de la base utilisateurs d'Echo en Afrique de l'Ouest.

L'approche combine un dataset NSFW généraliste à grande échelle pour la détection fondamentale de nudité, avec des datasets de diversité ciblés pour éliminer les faux positifs corrélés à la densité de mélanine.

---

## 1. Datasets de diversité (correction de biais)

Ces datasets servent à calibrer la frontière Safe-For-Work (SFW) du modèle, en s'assurant que les peaux foncées, les environnements à faible luminosité et les ombres à fort contraste ne sont pas classifiés à tort comme contenu explicite.

### Casual Conversations v2 (Meta AI)

- **Description :** Dataset à grande échelle créé spécifiquement pour la correction de biais en IA. Contient des données vidéo et image de personnes de toutes origines géographiques, annotées selon l'échelle de Fitzpatrick. Les carnations foncées y sont massivement représentées.
- **Source :** Meta AI Research (open-source)

### FairFace Dataset

- **Description :** Dataset d'images de visages explicitement équilibré pour la diversité ethnique (Black, White, Latino, Asian, Indian, Middle Eastern).
- **Application :** Entraine le classifieur à reconnaitre que les visages et cous à peau foncée dans des environnements à faible luminosité (ex : chambre peu éclairée) constituent du contenu safe.
- **Source :** Hugging Face / GitHub

### Diversity in Faces (DiF) Dataset (IBM)

- **Description :** Dataset de plusieurs millions d'images axé sur la diversité des traits humains et de la distribution de mélanine.
- **Source :** IBM Research

---

## 2. Datasets de contenu explicite (NSFW)

Ces datasets fournissent le signal d'entrainement fondamental pour la détection de nudité anatomique : géométrie corporelle, zones anatomiques exposées et classification de poses.

### deepghs/nsfw_detect

- **Description :** Dataset de classification d'images activement maintenu, adapté au fine-tuning direct.
- **Source :** Hugging Face Datasets

### DarkyMan/nsfw-image-classification

- **Description :** Dataset multi-classes (Sexy, Porn, Neutral). Les catégories "Sexy" et "Porn" sont fusionnées en une seule classe NSFW pour la classification binaire.
- **Source :** Hugging Face

### Yahoo NSFW as MobileNetV2 Bottlenecks

- **Description :** Sorties pré-calculées de modèles sur des milliers d'images NSFW. Adapté au transfer learning accéléré sur architectures MobileNet.
- **Source :** Kaggle

---

## 3. Méthodologie d'entrainement

### Phase 1 : Équilibrage du dataset (règle du 50/50)

Le répertoire d'entrainement doit imposer une parité démographique stricte :

```text
dataset/
├── train/
│   ├── SFW/   50% images population générale + 50% FairFace/CasualConversations (peaux foncées)
│   └── NSFW/  Datasets Hugging Face + augmentation ColorJitter (assombrissement artificiel)
```

Cette structure force le réseau à apprendre les caractéristiques structurelles de la nudité indépendamment de la couleur de peau.

### Phase 2 : Hard Negative Mining

Après l'entrainement initial, le modèle est affiné itérativement par ré-entrainement ciblé :

1. Identifier les cas de faux positifs (ex : un utilisateur torse nu dans une pièce sombre incorrectement flaggé NSFW).
2. Ajouter ces images dans la partition SFW avec les labels corrects.
3. Ré-entrainer le modèle sur le dataset augmenté.

Ce processus itératif, répété sur 2 à 3 cycles avec des images localement représentatives, produit un classifieur à la fois léger et contextuellement adapté à la démographie des utilisateurs d'Echo.

---

## 4. Sources des datasets

| Dataset | URL |
|:--------|:----|
| deepghs/nsfw_detect | https://huggingface.co/datasets/deepghs/nsfw_detect |
| Casual Conversations v2 | https://ai.meta.com/datasets/casual-conversations-v2-dataset/ |
| HuggingFaceM4/FairFace | https://huggingface.co/datasets/HuggingFaceM4/FairFace |
| IBM Diversity in Faces | https://www.research.ibm.com/artificial-intelligence/trusted-ai/diversity-in-faces/ |
| DarkyMan/nsfw-image-classification | https://huggingface.co/datasets/DarkyMan/nsfw-image-classification |
| Yahoo NSFW Bottlenecks | https://www.kaggle.com/datasets/nmurray1234/yahoo-nsfw-as-mobilenetv2-bottlenecks |
