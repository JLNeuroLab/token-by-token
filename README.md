# LLM Project

---

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
📦 LLM_project/                       Root of the LLM project
├──── 📂 configs/                      Configuration folder
│     ├──── 📄 base_config.py          Base configurations (e.g., general parameters)
│     └──── 📄 gpt_config.py           GPT-specific configurations
├──── 📂 data/                         Data folder
│     ├──── 📂 processed/              Preprocessed data
│     └──── 📂 raw/                    Raw/original data
├──── 📂 experiments/                  Scripts or notes for experiments
├──── 📂 llm_project/                  Main project code
│     ├──── 📂 bpe/                    Byte Pair Encoding (tokenization)
│     │     └──── 📄 bytepair_encoding.py   BPE implementation script
│     ├──── 📂 models/                 Implemented models
│     │     ├──── 📂 gpt/              GPT model
│     │     │     ├──── 📄 attention.py     Attention module
│     │     │     ├──── 📄 generator.py     Text generation module
│     │     │     ├──── 📄 model.py         GPT architecture definition
│     │     │     └──── 📄 train.py         GPT training loop
│     │     ├──── 📂 neural_ngrams/        Neural n-gram model
│     │     │     ├──── 📄 model.py         Neural n-gram network definition
│     │     │     └──── 📄 trainer.py       Neural n-gram training loop
│     │     └──── 📂 ngrams/               Classic n-gram model
│     │           ├──── 📄 model.py         N-gram definition
│     │           └──── 📄 trainer.py       N-gram training loop
│     └──── 📂 utils/                      Utility functions
│           ├──── 📄 dataloader.py         Data loading and batching
│           ├──── 📄 file_manager.py       File handling (read/write)
│           └──── 📄 token_mapping.py      Token <-> index mapping
├──── 📂 tests/                           Automated tests
│     ├──── 📄 test_attention.py          Tests for attention module
│     ├──── 📄 test_bpe.py                Tests for BPE
│     ├──── 📄 test_generator.py          Tests for generator module
│     ├──── 📄 test_model.py              Tests for models
│     └──── 📄 test_training_integration.py  Integration tests for training
├──── 📄 experiments.txt                  Notes/results from experiments
├──── 📄 main.py                          Main entry point to run the project
├──── 📄 README.md                        Project documentation
├──── 📄 requirements.txt                 Project dependencies
└──── 📄 run_experiments.py              Script to run specific experiments

```
## 3. How to use
```