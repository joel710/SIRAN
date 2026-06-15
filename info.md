# SIRAN - Training Data Strategy & Bias Mitigation Reference

## Objective

Correct demographic bias in the MobileNetV3 classifier to ensure uniform detection accuracy across all skin tones, with particular emphasis on dark-skinned populations representative of the Echo user base in West Africa.

The approach combines a large-scale generalist NSFW dataset for foundational nudity detection with targeted diversity datasets to eliminate false positives correlated with melanin density.

---

## 1. Diversity Datasets (Bias Correction)

These datasets serve to calibrate the model's Safe-For-Work (SFW) boundary, ensuring that dark skin tones, low-light environments, and high-contrast shadows are not erroneously classified as explicit content.

### Casual Conversations v2 (Meta AI)

- **Description:** Large-scale dataset created specifically for AI bias correction. Contains video and image data of individuals across all geographic origins, annotated with Fitzpatrick skin type classifications. Dark skin tones are heavily represented.
- **Source:** Meta AI Research (open-source)

### FairFace Dataset

- **Description:** Face image dataset explicitly balanced for ethnic diversity (Black, White, Latino, Asian, Indian, Middle Eastern).
- **Application:** Trains the classifier to recognize that dark-skinned faces and necks in low-light environments (e.g., dimly lit rooms) constitute safe content.
- **Source:** Hugging Face / GitHub

### Diversity in Faces (DiF) Dataset (IBM)

- **Description:** Multi-million image dataset focused on diversity of human traits and melanin distribution.
- **Source:** IBM Research

---

## 2. Explicit Content Datasets (NSFW)

These datasets provide the foundational training signal for anatomical nudity detection: body geometry, exposed anatomical regions, and pose classification.

### deepghs/nsfw_detect

- **Description:** Actively maintained image classification dataset suitable for direct fine-tuning.
- **Source:** Hugging Face Datasets

### DarkyMan/nsfw-image-classification

- **Description:** Multi-class image dataset (Sexy, Porn, Neutral). The "Sexy" and "Porn" categories are merged into a single NSFW class for binary classification.
- **Source:** Hugging Face

### Yahoo NSFW as MobileNetV2 Bottlenecks

- **Description:** Pre-computed model outputs on thousands of NSFW images. Suitable for accelerated transfer learning on MobileNet architectures.
- **Source:** Kaggle

---

## 3. Training Methodology

### Phase 1: Dataset Balancing (50/50 Rule)

The training directory must enforce strict demographic parity:

```text
dataset/
├── train/
│   ├── SFW/   50% general-population images + 50% FairFace/CasualConversations (dark skin tones)
│   └── NSFW/  Hugging Face datasets + ColorJitter augmentation (artificial darkening)
```

This structure forces the network to learn structural features of nudity independently of skin color.

### Phase 2: Hard Negative Mining

After initial training, the model is iteratively refined through targeted retraining:

1. Identify false positive cases (e.g., a shirtless user in a dark room incorrectly flagged as NSFW).
2. Add these images to the SFW partition with correct labels.
3. Retrain the model on the augmented dataset.

This iterative process, repeated over 2-3 cycles with locally representative imagery, produces a classifier that is both lightweight and contextually adapted to the Echo user demographic.

---

## 4. Dataset Sources

| Dataset | URL |
|:--------|:----|
| deepghs/nsfw_detect | https://huggingface.co/datasets/deepghs/nsfw_detect |
| Casual Conversations v2 | https://ai.meta.com/datasets/casual-conversations-v2-dataset/ |
| HuggingFaceM4/FairFace | https://huggingface.co/datasets/HuggingFaceM4/FairFace |
| IBM Diversity in Faces | https://www.research.ibm.com/artificial-intelligence/trusted-ai/diversity-in-faces/ |
| DarkyMan/nsfw-image-classification | https://huggingface.co/datasets/DarkyMan/nsfw-image-classification |
| Yahoo NSFW Bottlenecks | https://www.kaggle.com/datasets/nmurray1234/yahoo-nsfw-as-mobilenetv2-bottlenecks |
