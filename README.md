# Language Model from Scratch

A beginner-friendly Jupyter Notebook that builds a text generation language model from scratch using an LSTM neural network and TensorFlow/Keras.

---

## Overview

This notebook walks through the full pipeline of building a next-word prediction model:

1. Creating and saving a custom text dataset
2. Tokenizing and encoding text
3. Generating n-gram input/output sequences
4. Training an Embedding + LSTM model
5. Generating new text from a seed phrase

---

## Demo

**Seed:** `"Language modeling"`  
**Generated:** `"Language modeling is fascinating and neural networks can learn"`

---

## Requirements

Install dependencies before running:

```bash
pip install numpy==1.26.4 tensorflow==2.17.1 keras==3.5.0
```

Or install the latest compatible versions:

```bash
pip install numpy tensorflow keras
```

> If you encounter version conflicts, use the pinned versions above.

---

## How It Works

### 1. Dataset
A custom dataset of ~90 AI/ML-related sentences is used for training. The data is saved to `custom_data.txt`.

### 2. Tokenization
Text is tokenized using Keras `Tokenizer`, which builds a vocabulary and maps each word to an integer index.

### 3. Sequence Preparation
N-gram sequences are generated from each sentence. For example:
```
"Language modeling is"
→ [Language modeling] → is
→ [Language modeling is] → fascinating
```
Sequences are padded to a uniform length.

### 4. Model Architecture

| Layer | Details |
|---|---|
| `Embedding` | `vocab_size × 128` — maps word indices to dense vectors |
| `LSTM` | 256 units — captures sequential patterns |
| `Dense` | `vocab_size` units, softmax activation — predicts next word |

- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy  
- **Epochs:** 100  
- **Batch size:** 32

### 5. Text Generation
Given a seed phrase, the model predicts one word at a time, appending each predicted word to the seed until the desired length is reached.

```python
generate_text(model, tokenizer, seed_text="Language modeling", max_length=10)
```

---

## File Structure

```
├── Creating_a_language_model_from_scratch.ipynb   # Main notebook
├── custom_data.txt                                # Auto-generated training data
└── README.md
```

---

## Notebook Steps

| Step | Description |
|---|---|
| Example Dataset | Defines ~90 AI/ML sentences as training data |
| Save Dataset | Writes sentences to `custom_data.txt` |
| Import Libraries | Imports NumPy, TensorFlow, Keras |
| Load Dataset | Reads `custom_data.txt` |
| Tokenize | Fits `Tokenizer` on the corpus |
| Build Sequences | Creates n-gram input/output pairs |
| Pad Sequences | Pads all sequences to uniform length |
| Categorical Targets | One-hot encodes target words |
| Define Model | Builds Embedding → LSTM → Dense |
| Train | Trains for 100 epochs |
| Generate Text | Runs inference from a seed phrase |

---

## Notes

- Training on the small built-in dataset is fast but produces limited variety. Swap in a larger corpus via `custom_data.txt` for better results.
- The model uses greedy decoding (`argmax`). For more creative output, replace with temperature sampling or top-k sampling.
- GPU is recommended for faster training but not required.

---

## License

MIT License

Copyright (c) 2024 IMApurbo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Author

**IMApurbo**  
[github.com/IMApurbo](https://github.com/IMApurbo)
