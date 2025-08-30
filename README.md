# Token by Token
__Token by Token__ is a project focused on studying and experimenting with token-level language models. We implemented three types of models: the classic __n-gram__, the __neural n-gram__, and a __GPT-based model__. All models were trained on the __Shakespeare dataset__. Model evaluation was conducted both quantitatively, using __perplexity__, and qualitatively, through __text generation__, assessing each model’s ability to produce coherent sequences faithful to the original style.

## Table of Contents
1. [Environment Setup](#environment-setup)
   1. [Clone the repository](#11-clone-the-repository)
   2. [Create virtual environment](#12-create-virtual-environment)
2. [Project Structure](#2-project-structure)
3. [How to Use](#3-how-to-use)

---

## 1 Environment Setup

### 1.1 Clone the repository
```
git clone <https://github.com/JLNeuroLab/token-by-token.git>
```
### 1.2 Create virtual environment
```
python -m venv venv
```
### 1.3 Activate virtual environment
```
.\venv\Scripts\Activate.ps1
```

### 1.4 Install dependencies
```
# Make sure the venv is activated

pip install -r requirements.txt
```

### 1.5 Test setup
```
python main.py
```

## 2 Project Structure
```
📦 LLM_project/
├────── 📂 configs/
├────── 📂 data/
│       ├────── 📂 processed/
│       └────── 📂 raw/
├────── 📂 experiments/
├────── 📂 llm_project/
│       ├────── 📂 bpe/
│       ├────── 📂 models/
│       │       ├────── 📂 gpt/
│       │       ├────── 📂 neural_ngrams/
│       │       └────── 📂 ngrams/
│       └────── 📂 utils/
├────── 📂 tests/
├────── 📄 experiments.txt
├────── 📄 pyproject.toml
├────── 📄 README.md
└────── 📄 requirements.txt
## 3. How to use
```