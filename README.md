# Angular Supervision for Visual Object Grounding in Autoregressive VLMs

**Parsa Esmaeilkhani, Longin Jan Latecki** · Temple University

[[Paper (arXiv:2610.32220)]](https://arxiv.org/abs/2610.32220)

> **Note:** The code is to be completed.

## Installation

```bash
git clone https://github.com/ParsaKhani/Angular_object_grounding.git
cd Angular_object_grounding
pip install -r requirements.txt
```

Tested with torch 2.5.1, transformers 5.10, peft 0.18, deepspeed 0.18.

## Data

Training data is a JSON list in LLaVA format. Image paths are relative to `--image_folder`:

```json
[
  {
    "image": "train2014/COCO_train2014_000000581857.jpg",
    "conversations": [
      {"from": "human", "value": "<image> Please provide the bounding box coordinate of the region this sentence describes: the lady with the blue shirt."},
      {"from": "gpt", "value": "[103.93, 299.99, 238.15, 477.41]"}
    ]
  }
]
```

Only the assistant turns are supervised.

## Training

```bash
bash scripts/train_qwen2_5_vl.sh
```

Edit the variables at the top of the script to set the data paths, the number of GPUs and the hyperparameters.
LoRA is applied to the language model, the vision-to-language projector is fully trained, and the vision encoder is frozen.
The LoRA adapter (with the projector weights) is saved to `OUTPUT_DIR`.

## Citation

```bibtex
@article{esmaeilkhani2026angular,
  title   = {Angular Supervision for Visual Object Grounding in Autoregressive Vision-Language Models},
  author  = {Esmaeilkhani, Parsa and Latecki, Longin Jan},
  journal = {arXiv preprint arXiv:2610.32220},
  year    = {2026}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
