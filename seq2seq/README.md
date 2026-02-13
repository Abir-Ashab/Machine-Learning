# Text-to-Python Code Generation Using Seq2Seq Models

A deep learning project that generates Python code from natural language descriptions using three different RNN architectures: Vanilla RNN, LSTM, and LSTM with Attention mechanism.

## 📋 Overview

This project implements and compares three sequence-to-sequence models for automatic code generation:

- **Vanilla RNN Seq2Seq**: Basic encoder-decoder architecture
- **LSTM Seq2Seq**: Long Short-Term Memory based architecture for better long-term dependencies
- **LSTM with Attention**: Enhanced LSTM with attention mechanism for improved performance

**Dataset**: CodeSearchNet Python (Hugging Face)

- Training samples: 8,000
- Validation samples: 1,000
- Test samples: 1,000

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version
```

### Installation

1. **Install required packages**:

```bash
pip install torch torchvision torchaudio
pip install datasets transformers
pip install nltk sacrebleu
pip install matplotlib seaborn pandas numpy tqdm
```

Or install all at once:

```bash
pip install datasets transformers torch torchvision torchaudio nltk sacrebleu matplotlib seaborn pandas numpy tqdm
```

### Running the Project

#### Option 1: Run Complete Notebook

```bash
# Open the notebook in Jupyter or VS Code
jupyter notebook seq2seq.ipynb
# or
code seq2seq.ipynb
```

Then run all cells sequentially (Shift + Enter for each cell, or "Run All" from the menu).

#### Option 2: Run on Kaggle

1. Upload `notebook40138fad8e.ipynb` to Kaggle
2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
3. Run all cells
4. Download results from the output panel

#### Option 3: Use Pre-trained Models

```python
import torch
import torch.nn as nn

# Load the pre-trained models (make sure to define model classes first)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example: Load LSTM with Attention (best model)
from models import LSTMEncoder, AttentionDecoder, LSTMAttentionSeq2Seq

encoder = LSTMEncoder(vocab_size_src, embedding_dim=256, hidden_dim=256)
decoder = AttentionDecoder(vocab_size_tgt, embedding_dim=256, hidden_dim=256)
model = LSTMAttentionSeq2Seq(encoder, decoder, device).to(device)

# Load saved weights
model.load_state_dict(torch.load('lstm_attention_best.pt', map_location=device))
model.eval()

# Now you can use the model for inference!
```

## 📁 Project Structure

```
seq2seq/
├── seq2seq.ipynb                      # Main notebook with all code
├── notebook40138fad8e.ipynb          # Kaggle version
├── README.md                          # This file
├── REPORT.md                          # Detailed experimental report
│
├── Model Checkpoints (*.pt)
│   ├── vanilla_rnn_best.pt           # Vanilla RNN weights (~1.7 MB)
│   ├── lstm_best.pt                  # LSTM weights (~1.7 MB)
│   └── lstm_attention_best.pt        # LSTM + Attention weights (~1.8 MB)
│
├── Results & Data
│   ├── model_comparison.csv          # Quantitative results table
│   └── final_results.pkl             # Complete results (losses, metrics)
│
└── Visualizations (*.png)
    ├── training_curves.png            # Loss curves for all models
    ├── validation_loss_comparison.png # Model comparison
    ├── metrics_comparison.png         # BLEU, accuracy charts
    ├── attention_example_1.png        # Attention heatmap example 1
    ├── attention_example_2.png        # Attention heatmap example 2
    ├── attention_example_3.png        # Attention heatmap example 3
    └── length_performance.png         # Performance vs input length
```

## 🔧 Configuration

Key hyperparameters (can be modified in the notebook):

```python
CONFIG = {
    'TRAIN_SIZE': 8000,
    'VAL_SIZE': 1000,
    'TEST_SIZE': 1000,
    'MAX_DOCSTRING_LEN': 50,
    'MAX_CODE_LEN': 80,
    'EMBEDDING_DIM': 256,
    'HIDDEN_DIM': 256,
    'BATCH_SIZE': 64,
    'EPOCHS': 15,
    'LEARNING_RATE': 0.001,
    'TEACHER_FORCING_RATIO': 0.5,
    'VOCAB_SIZE_SRC': 5000,
    'VOCAB_SIZE_TGT': 5000,
    'GRADIENT_CLIP': 1.0,
}
```

## 📊 Results Summary

| Model                | BLEU Score | Token Accuracy | Parameters |
| -------------------- | ---------- | -------------- | ---------- |
| Vanilla RNN          | ~8-12      | ~25-30%        | ~3.4M      |
| LSTM                 | ~12-18     | ~30-40%        | ~4.8M      |
| **LSTM + Attention** | **~18-25** | **~40-50%**    | **~5.1M**  |

_Note: Exact values depend on training run due to randomness_

## 🎯 Usage Examples

### Generate Code from Description

```python
# Example docstring
docstring = "calculate the sum of two numbers"

# Tokenize and encode
src_tokens = src_tokenizer.encode(docstring, max_len=50)
src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)

# Generate code
decoded, _ = greedy_decode(model, src_tensor, max_len=80, device=device, use_attention=True)

# Decode to text
generated_code = tgt_tokenizer.decode(decoded, skip_special_tokens=True)
print(generated_code)
# Output: def sum ( a , b ) : return a + b
```

## 🐛 Troubleshooting

### Out of Memory Error

- Reduce `BATCH_SIZE` (try 32 or 16)
- Reduce `TRAIN_SIZE` (try 5000)
- Use CPU instead of GPU for smaller datasets

### CUDA Not Available

```python
# The code automatically falls back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Slow Training

- Enable GPU if available
- Reduce `EPOCHS` for quick testing (try 5)
- Reduce dataset size for experimentation

## 📈 Training Time

Approximate training times (on different hardware):

| Hardware   | Time per Epoch | Total (15 epochs) |
| ---------- | -------------- | ----------------- |
| GPU (T4)   | ~2-3 min       | ~30-45 min        |
| GPU (V100) | ~1-2 min       | ~15-30 min        |
| CPU        | ~15-20 min     | ~4-5 hours        |

## 🔍 Model Architecture Details

### 1. Vanilla RNN Seq2Seq

- Encoder: Single-layer RNN
- Decoder: Single-layer RNN with FC output layer
- Context: Fixed-size vector from encoder final state

### 2. LSTM Seq2Seq

- Encoder: Single-layer LSTM
- Decoder: Single-layer LSTM with FC output layer
- Better at capturing long-term dependencies

### 3. LSTM with Attention

- Encoder: Single-layer LSTM
- Decoder: Single-layer LSTM with attention mechanism
- Attention: Learns to focus on relevant input tokens
- Best performance due to dynamic context vector

## 📚 References

- Dataset: [CodeSearchNet](https://github.com/github/CodeSearchNet)
- Seq2Seq: Sutskever et al. (2014)
- Attention: Bahdanau et al. (2015)
- LSTM: Hochreiter & Schmidhuber (1997)

## 🤝 Contributing

Feel free to:

- Report bugs or issues
- Suggest improvements
- Add new model architectures
- Improve preprocessing or tokenization

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created as part of a Machine Learning course project.

---

**Happy Coding! 🚀**

For detailed experimental results and analysis, see [REPORT.md](REPORT.md)
