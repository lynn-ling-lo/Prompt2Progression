# From Prompt to Progression: Taming Video Diffusion Models for Seamless Attribute Transition (ICCV2025)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Abstract:** Existing models often struggle with complex temporal changes, particularly when generating videos with gradual attribute transitions. The most common prompt interpolation approach for motion transitions often fails to handle gradual attribute transitions, where inconsistencies tend to become more pronounced. In this work, we propose a simple yet effective method to extend existing models for smooth and consistent attribute transitions, through introducing frame-wise guidance during the denoising process. Our approach constructs a data-specific transitional direction for each noisy latent, guiding the gradual shift from initial to final attributes frame by frame while preserving the motion dynamics of the video. Moreover, we present the Controlled-Attribute-Transition Benchmark (CAT-Bench), which integrates both attribute and motion dynamics, to comprehensively evaluate the performance of different models. We further propose two metrics to assess the accuracy and smoothness of attribute transitions. Experimental results demonstrate that our approach performs favorably against existing baselines, achieving visual fidelity, maintaining alignment with text prompts, and delivering seamless attribute transitions.

[arXiv](https://arxiv.org/abs/2509.19690)

## 🎯 Highlights

![Teaser](assets/teaser.png)
*Example of Video Generation with Attribute Transitions Using the Same Base Model. The base model generates static appearances throughout the video. Prompt interpolation leads to inconsistencies, such as abrupt changes in the buildings, while our method ensures smoother and more consistent attribute transitions.*

**Key Contributions:**
- 🚀 **Prompt2Progress**: A novel method for text-to-video generation that improves temporal consistency for attribute transition without further training
- 📊 **CATbench**: A comprehensive benchmark for evaluating temporal consistency in video generation with attribute transition on two novel metrics: Wholistic Transition Score and Frame-wise Transition Score
- 🔧 **Extensive Evaluation**: Systematic comparison using diverse prompts across multiple attributes on video diffusion models.

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/lynn-ling-lo/Prompt2Progression.git
cd Prompt2Progression

# Create conda environment
conda create -n p2p python=3.8.5
conda activate p2p

# Install dependencies
pip install -r requirements.txt
```

### Quick Demo

#### Prompt2Progress Text-to-Video Generation with Attribute Transition
```bash
cd prompt2progress

# Download pretrained VideoCrafter2 weights
# Put the model.ckpt in checkpoints/base_512_v2/model.ckpt
wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt -P checkpoints/base_512_v2/
# Or manually download and place model.ckpt in checkpoints/base_512_v2/

# Run text-to-video generation
sh scripts/run_text2video_mp.sh
```

#### CATbench Evaluation
```bash
cd CATbench

# Evaluate temporal consistency using our metrics
python evaluate.py --videos_path /path/to/videos \
                   --prompt_file /path/to/prompt/file \
```

## 📊 Results

### Quantitative Results
| Method | Wholistic Transition Score ↑ | Frame-wise Transition Score ↑ | 
|--------|-------|-------|
| AnimateDiff | 0.0082 | 0.0004 |
| Modelscope | 0.0042 | 0.0001 |
| Latte | 0.0019 | -0.0002 |
| VideoCrafter2 | 0.0022 | 0.0003|
| Free-Bloom | 0.1077 | -0.0020 |
| VideoTetris |0.0134 | 0.0012 |
| Gen-L | 0.1166 | 0.0135 |
| FreeNoise |0.0578 | 0.0066 |
| **Ours** | **0.1486** | **0.0201** |

### Qualitative Results




## 🛠️ Usage
### Prompt File Format
Both Prompt2Progression inference and CATbench evaluation use the same prompt format:
```
[initial state prompt];[final state prompt];[neutral prompt]
```

**Example (`age.txt`):**
```
a young girl is rowing a boat; an old girl is rowing a boat; a girl is rowing a boat
a young man walking in the park; an old man walking in the park; a man walking in the park  
a young woman reading a book; an old woman reading a book; a woman reading a book
```

**Example (`weather.txt`):**
```
a house in sunny weather; a house in rainy weather; a house
a car driving in clear sky; a car driving in stormy weather; a car driving
a garden on a bright day; a garden on a cloudy day ;a garden
```

**Prompt Guidelines:**
- Use semicolons (`;`) to separate the three components
- **Initial state**: Starting condition/attribute  
- **Final state**: Target condition/attribute to transition to
- **Neutral prompt**: Base prompt without specific attributes
- Keep consistent actions/objects across all three prompts
- One prompt triplet per line

### Prompt2Progress Inference

Generate videos with temporal consistency using your prompts:

```bash
cd prompt2progress

# 1. Configure hyperparameters in the script
# Edit scripts/run_text2video_mp.sh to set:
# - Input prompt file path
# - Output directory

# 2. Run inference
sh scripts/run_text2video_mp.sh
```

### CATbench Evaluation

Evaluate temporal consistency of generated videos:

```bash
cd CATbench

# Evaluate with custom prompts
python evaluate.py --prompt_file custom_prompts.txt \
                   --videos_path /path/to/your/videos \

```



## 📄 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{lo2025p2p,
  title={From Prompt to Progression: Taming Video Diffusion Models for Seamless Attribute Transition},
  author={Lo, Ling and Chan, Kelvin CK and Cheng, Wen-Huang and Yang Ming-Hsuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## 🙏 Acknowledgments
We extend our heartfelt gratitude to the open-source community whose foundational work enabled this research. We particularly thank:
- **[VideoCrafter](https://github.com/AILab-CVC/VideoCrafter)**

---

<div align="center">

**⭐ If this work is helpful for your research, please consider giving us a star! ⭐**

</div>
