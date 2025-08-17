# MTCL-IT: Multi-Task Contrastive Learning with Instruction Tuning for Metaphor Sentiment Analysis

**Authors:** Anonymous Authors


<!-- [[论文PDF]()] [[arXiv]()] [[项目主页]()] -->

<!-- ![Teaser Figure](docs/teaser.png) -->

>This repository contains the official implementation and experimental resources for the paper **[MTCL-IT: Multi-Task Contrastive Learning with Instruction Tuning for Metaphor Sentiment Analysis]**

<!-- ## 📝 Abstract -->



## 🗂 Repository Structure
```angular2html
├── 📁 checkpoints/          # Save the enhanced pre-trained Metaphor-BERT
├── 📁 data/                 # Datasets
├── 📁 docs/                 # Documentation & figures
└── 📁 src/                  # Core implementations
    ├── 📁 instruct/         # KNN-based intruction construction
    ├── 📁 models/           # Core models
    ├── 📁 utils/            # Data process & evaluation & some utils
    ├── 📄 main.py           # Main excution script
    └── 📄 pretrain.py       # Enhanced pre-trained for BERT yields Metaphor-BERT
```

## 🛠 Installation

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- CUDA ≥ 11.8

### Setup
```bash
# Clone repository
git clone https://github.com/XXXX/xxxxx.git
cd MTCL-IT

# Create conda environment
conda create -n msa python=3.8
conda activate msa

# Install dependencies
pip install -r requirements.txt
```
## 🚀 Usage
- **Enhanced Pre-training**
```bash
# Please modify the data and model paths
python pretrain.py
```

- **Training & Testing**

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

<!-- ## 📊 Results -->
<!-- Performance on [Dataset Name]:

Method	Metric1	Metric2	Metric3
Our Method	0.92	1.23	95.4%
Baseline	0.85	1.45	89.7%
Results Comparison -->

<!-- ## 📜 Citation -->
<!-- ```bash
@article{yourcitationkey,
  title     = {Your Paper Title},
  author    = {Author1 and Author2 and Author3},
  journal   = {Conference or Journal Name},
  year      = {2023}
}
``` -->


<!-- ## 📧 联系方式 -->
<!-- Corresponding Author: [Your Name] - your.email@institution.edu -->
