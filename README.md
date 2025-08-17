# Joint Modeling of Source, Target, and Context for Metaphor Sentiment Analysis: Dataset and Method

<!-- **Authors:** Anonymous Authors -->


<!-- [[论文PDF]()] [[arXiv]()] [[项目主页]()] -->

<!-- ![Teaser Figure](docs/teaser.png) -->

>This repository contains the official implementation and experimental resources for the paper **[Joint Modeling of Source, Target, and Context for Metaphor Sentiment Analysis: Dataset and Method]**

## 📝 Abstract
Metaphorical expressions evoke stronger emotions than literal language. However, their implicit sentiment mapping characteristics pose significant challenges for sentiment analysis. Existing studies exhibit two major limitations: first, most models focus on modeling the partial of three core elements (the source-domain word $S$, the target-domain word $T$, and the metaphorical context $C$), neglecting the interactions among them in metaphors; second, current research in computational metaphor predominantly adopts a single-type metaphor pattern, making it difficult for models to capture shared representations across different metaphor types, thereby limiting their generalization. To address these issues, this study proposes a multi-type metaphor sentiment analysis framework that jointly models $S$, $T$, and $C$, aiming to accurately identify the sentiment of complex metaphorical texts while revealing the path of metaphorical sentiment transfer, thereby facilitating sentiment analysis of metaphor-rich texts. To effectively model the three elements, a metaphor sentiment analysis model integrating Multi-Task Contrastive Learning and Instruction Tuning (MTCL-IT) is proposed. Finally, multi-perspective experiments are conducted on the constructed English dataset EMSA and Chinese dataset CMSA, both containing $S$, $T$, sentiment polarity, and metaphor type information, fully validating the effectiveness of the proposed framework and model in sentiment recognition and understanding of sentiment mapping mechanisms.



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
