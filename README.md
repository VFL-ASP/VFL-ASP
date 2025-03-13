# VFL-ASP: Vertical Federated Learning Across Second-hop Parties

**VFL-ASP** is a novel framework designed to enhance Vertical Federated Learning (VFL) by integrating feature information from **second-hop passive parties** during training and inference. This method addresses limitations in traditional VFL systems that require full data overlap between active and passive parties.

---

## 🔥 Key Features
- **Second-hop Feature Integration**: Utilizes embeddings from non-overlapping second-hop passive parties for improved model performance.
- **Efficient Federated SVD**: Introduces a single encryption matrix approach to enhance computation efficiency.
- **Knowledge Distillation**: Trains a student model for independent inference at the active party, reducing dependency on overlapping data.
- **Improved Accuracy**: Outperforms baseline VFL methods in diverse data configurations.

---

## 🧩 VFL-ASP Framework Modules
VFL-ASP operates through the following four modules:
1. **Embedding Extraction**: Extracts hidden embeddings using Federated SVD from overlapping data between first-hop and second-hop passive parties.
2. **Embedding Approximation**: Uses an autoencoder for semi-supervised learning to approximate extracted embeddings for non-overlapping samples.
3. **Teacher Model**: Utilizes the approximated embeddings and active party data to construct the VFL system, generating both predictions and soft labels.
4. **Student Model**: A lightweight model at the active party that performs inference independently with guidance from the teacher model’s soft labels.

---

## 📁 Project Structure
```
├── configs/                # Configuration files for model training
├── dataset/                # Datasets used for model evaluation
├── main.py                 # Main script to run the VFL-ASP framework
├── fsvd.py                 # Federated Singular Value Decomposition
├── train_fsvd.py           # Federated SVD training script
├── ae.py                   # Autoencoder for embedding approximation
├── train_ae.py             # Autoencoder training script
├── vfl_kd.py               # VFL system + Knowledge distillation implementation
├── utils.py                # Utility functions for data processing
├── performance.py          # Performance metrics for ablation study
├── feature_select.py       # Feature selection methods for second-hop data
```

---

## ⚙️ Installation
### 1. Requirements
```
jax==0.4.20
jaxlib==0.4.20+cuda11.cudnn86
numpy==1.24.4
pandas==1.5.3
scikit-learn==1.5.1
scipy==1.10.1
flax==0.8.5
optax==0.2.4
torch==2.3.1
tensorstore==0.1.69
```
### 2. Setup
```bash
git clone https://github.com/VFL-ASP/VFL-ASP.git
cd VFL-ASP
pip install -r requirements.txt
```

---

## 📊 Results
| Model    | Breast Cancer | MIMIC-III | Credit |
|-----------|----------------|-------------|---------|
| VFL-ASP    | **93.03%**      | 59.93%       | **78.41%** |
| VFL-STD    | 91.64%          | **61.83%**   | 78.36%      |
| LOCAL | 87.55%          | 44.16%       | 76.75%      |

---
