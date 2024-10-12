# Multi-Guard



Multi-Guard: A Multimodal Large Language Model Framework for Detecting Unsafe Content

Multi-Guard is a versatile framework for evaluating the safety compliance of multimodal content. It is designed to review and safeguard both textual and visual content, supporting dataset annotation and ensuring the safety of generative models.

# Dataset and weights



The dataset used in this study is available for download via the following link: https://pan.baidu.com/s/1n74InQg0IyEWJ-FETrEHYw (Extraction code: accd) or https://huggingface.co/datasets/waterpeople/Multi-Guard/tree/main. Due to the sensitive nature of the dataset, if the link becomes invalid or inaccessible, please contact the corresponding author to request access to the dataset.

The weights used in this study is available via the following link: https://huggingface.co/waterpeople/Multi-Guard/tree/main.and you also download llama-guard3 weight from https://github.com/meta-llama/llama-recipes

# Install Package



conda create -n multiguard python=3.10 -y

conda activate multiguard

pip install -e .

# Quick Start



the usage example is in the run_multiguard.py

You only need to modify the picture path(image_prompt) and the text(text_prompt).

## Training by yourself



Convert your data to a JSON file of a List of all samples. Sample metadata should contain `id` (a unique identifier), `image` (the path to the image), and `label` 

A sample JSON for finetuning Multi-Guard :

```json
[
  {
    "id": "sex001",
    "image": "part-000001/violence.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\ndescribe this picture."
      },
      {
        "from": "gpt",
        "value": "a man is killing a women."
      },
    ]
  },
  ...
]
```

Then, you can use `finetune-multiguard.sh` to fine-tune the multiguard.


Finally, you need to use `merge-model.sh` to merge the fine-tuned model.

