# DetRefiner Realtime Demo

Run DetRefiner on arbitrary images and save detection results before and after score refinement.

This demo computes features online:

- Text features: MobileCLIP-B
- Visual features: DINOv3 ViT-B/16
- Base detector: a Hugging Face open-vocabulary object detector

## Requirements

Install the required packages, including:

```bash
pip install torch torchvision transformers pillow numpy
```

Also install [MobileCLIP](https://github.com/apple/ml-mobileclip) and prepare the following model files:

- DetRefiner checkpoint
- MobileCLIP-B checkpoint
- DINOv3 ViT-B/16 model
- Hugging Face detector model

## Usage

### Image directory

```bash
python3 demo_detrefiner.py \
  --image-dir path/to/images \
  --class-names person dog bicycle \
  --detector-name llmdet-large \
  --detrefiner-ckpt trained_models/detrefiner_lvis.pth \
  --out-dir outputs_detrefiner
```

Use `--recursive` to include subdirectories.

### Single image

```bash
python3 demo_detrefiner.py \
  --image path/to/image.jpg \
  --class-names person dog bicycle \
  --detector-name llmdet-large \
  --detrefiner-ckpt trained_models/detrefiner_lvis.pth \
  --out-dir outputs_detrefiner
```

Specify exactly one of `--image` or `--image-dir`.

## Supported Detectors

- Grounding DINO Tiny / Base
- MM-Grounding DINO Tiny / Base / Large
- LLMDet Tiny / Base / Large

Use `--help` to see all options.

## Output

```text
outputs_detrefiner/
├── before_png/   # Original detector scores
├── after_png/    # Refined scores
├── concat_png/   # Before/after comparison
└── json/         # Bounding boxes and before/after scores
```

Each JSON file contains all candidates retained after detector candidate generation. Display thresholds, class-wise NMS, and `--topk` are applied only to the PNG visualizations.

## Notes

- Bounding boxes and class labels are not changed; DetRefiner only recalibrates confidence scores.
- `--candidate-score-thr 0` and `--max-dets -1` retain all detector candidates and may require substantial memory for large vocabularies.
- The default DetRefiner architecture expects DINOv3 ViT-B/16 features with 196 patch tokens and a hidden dimension of 768.

## Reference

For the full training and evaluation code, see the main DetRefiner repository and paper.
