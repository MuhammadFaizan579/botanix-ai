# BotaniX AI: Flower Species Classification

Identifies 102 flower species from a photo using transfer learning with MobileNetV2, and runs in the browser through a web app deployed on Vercel.

## Project links

| Item | Link |
|---|---|
| Live web app (Vercel) | https://botanix-flower-ai.vercel.app/ |
| Project report | https://botanix-flower-ai.vercel.app/Flower_Classification_Report.pdf |
| Presentation | https://botanix-flower-ai.vercel.app/Flower_Classification_Presentation.pdf |
| LinkedIn post |[ [https://lnkd.in/p/dqXdgjjH |
| Dataset | https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ |
| Training notebook | [notebooks/BotaniX_Flower_Classification.ipynb](notebooks/BotaniX_Flower_Classification.ipynb) |
| Model weights | [weights/](weights/) |
| Source code | [src/](src/) |

## Dataset

**Oxford 102 Flowers** (Visual Geometry Group, University of Oxford): 8,189 images, 102 flower species.
The dataset is not stored in this repository. It downloads automatically with torchvision:

```python
from torchvision import datasets
datasets.Flowers102(root="data", split="train", download=True)
```

The three official splits (train, val, test) were merged and re-split 70/15/15 with seed 42:
5,732 training, 1,228 validation and 1,229 test images.

## Model and training

- Architecture: MobileNetV2 pretrained on ImageNet, final layer replaced with a 102-class linear layer
- Training: 10 epochs, AdamW (learning rate 3e-4), cosine learning-rate schedule, batch size 32, cross-entropy loss
- Augmentation: random resized crop, horizontal flip, rotation (15 degrees), colour jitter
- Preprocessing: resize 256, centre crop 224, ImageNet mean/std normalisation
- Hardware: Google Colab, T4 GPU, about 8.4 minutes

## Results (test set, 1,229 unseen images)

| Metric | Value |
|---|---|
| Accuracy | 97.72% |
| Macro F1-score | 0.9725 |
| Misclassified | 28 of 1,229 |

Most errors are between look-alike species (for example petunia and tree mallow, or sword lily and canna lily).
Training curves are in `results/curves.png` and example predictions in `results/sample_predictions.png`.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── class_names.json            102 class names (index = model output index)
├── notebooks/
│   └── BotaniX_Flower_Classification.ipynb    full training notebook with outputs
├── src/
│   ├── predict.py              predict one image
│   └── evaluate.py             re-compute test accuracy and F1
├── weights/
│   ├── best_model.pth          PyTorch weights
│   └── model.onnx              ONNX export used by the web app
├── results/
│   ├── curves.png
│   └── sample_predictions.png
└── web/                        Vercel site (index.html, script.js, model.onnx, class_names.json)
```

## Run it yourself

```bash
pip install -r requirements.txt

# Predict one image
python src/predict.py path/to/flower.jpg

# Re-evaluate on the test split (downloads the dataset on first run)
python src/evaluate.py
```

`predict.py` prints the top 3 species, one per line, as `flower name: confidence%`.

## Web app

The site in `web/` loads `model.onnx` in the browser with ONNX Runtime Web and shows the top 3 predictions with confidence scores. Photos never leave the user's device.
To run it locally:

```bash
cd web
python -m http.server 8000
# open http://localhost:8000
```

On Vercel, set the project's **Root Directory** to `web`.
