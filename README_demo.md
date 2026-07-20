# DetRefiner Realtime Demo

Run DetRefiner on arbitrary images and save detection results before and after score refinement.

This demo computes features online:

- Text features: MobileCLIP-B
- Visual features: DINOv3 ViT-B/16
- Base detector: a Hugging Face open-vocabulary object detector.  See `MODEL_REGISTRY` in [`demo_detrefiner.py`](demo_detrefiner.py) 

## Requirements

The demo was tested with:

- Python 3.9.18
- CUDA 12.8
- PyTorch 2.4.1
- torchvision 0.19.0
- transformers 4.56.2
- Linux

A CUDA-enabled GPU is recommended.

Install the required packages:

```bash
pip install torch==2.4.1 torchvision==0.19.0
pip install transformers==4.56.2 pillow numpy
```

Install [MobileCLIP](https://github.com/apple/ml-mobileclip):

```bash
git clone https://github.com/apple/ml-mobileclip.git
cd ml-mobileclip
pip install -e .
cd ..
```

## Model Files

Download and place the required model files as follows:

```text
detrefiner/
├── data/
│   └── huggingface/
│       └── <organization>/
│           └── <detector-model>/
├── models/
│   ├── facebook/
│   │   └── dinov3-vitb16-pretrain-lvd1689m/
│   └── mobileclip_blt.pt
└── trained_models/
    └── detrefiner_lvis.pth
```

For example, when using `llmdet-large`, place the detector model under:

```text
data/huggingface/iSEE-Laboratory/llmdet_large/
```

### Download Links

| Model | Download |
|---|---|
| DINOv3 ViT-B/16 | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) |
| MobileCLIP-B | [apple/MobileCLIP-B-LT](https://huggingface.co/apple/MobileCLIP-B-LT) |
| DetRefiner checkpoint | [detrefiner_lvis.pth](https://huggingface.co/sokazaki/detrefiner/blob/main/trained_models/detrefiner_lvis.pth) |
| LLMDet Large | [iSEE-Laboratory/llmdet_large](https://huggingface.co/iSEE-Laboratory/llmdet_large) |

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
- Text prompts are limited to 256 tokens (`max_text_len=256`); excessively long prompts may be truncated.

## Reference

For the full training and evaluation code, see the [main README](README.md) and the paper.
