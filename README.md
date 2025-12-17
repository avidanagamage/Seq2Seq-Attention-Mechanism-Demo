# Seq2Seq Attention Mechanism Demo

A comprehensive NLP project demonstrating the impact of the **Attention Mechanism** on Sequence-to-Sequence (Seq2Seq) models. This project compares a baseline GRU Encoder-Decoder against an Attention-based model (Bahdanau Attention) on a character-level string reversal task.

It includes a **Streamlit** web application to visualize predictions and attention heatmaps interactively.

## 📂 Project Structure

```text
NLP Project/
├── data/                  # Dataset generation and utilities
│   ├── generate_dataset.py
│   ├── dataset_utils.py
│   └── char_vocab.json
├── models/                # Model architectures and training scripts
│   ├── baseline/          # Standard Seq2Seq (Encoder-Decoder)
│   └── attention/         # Seq2Seq with Bahdanau Attention
├── demo_app/              # Streamlit Web Application
│   ├── app.py
│   └── sample_inputs.json
└── visualization/         # Scripts to generate training heatmaps
    └── save_attention_heatmaps.py
```
## ⚙️ Setup & Training

Because large dataset files and model weights are not stored in this repository, you must generate the data and train the models locally before running the demo.

### 1. Generate Dataset
Run the generation script to create the synthetic string reversal dataset and the character vocabulary (`char_vocab.json`).
```bash
cd data
python generate_dataset.py
cd ..
```
### 2. Train Baseline Model
Trains the standard GRU Seq2Seq model (without attention). The model weights will be saved to (`models/baseline/baseline_weights.pth`)
```bash
cd models/baseline
python train_baseline.py
cd ../..
```
### 3. Train Attention Model
Trains the Bidirectional GRU model with Bahdanau Attention. This script saves the weights to (`models/attention/attention_weights.pth`) and logs attention matrices for visualization.
```bash
cd models/baseline
python train_baseline.py
cd ../..
```
### 4. Generate Training Visualizations (Optional)
Creates static heatmaps showing how the attention alignment evolved during training (from random noise to a diagonal).
```bash
python visualization/save_attention_heatmaps.py
```
## 🖥️ Running the Demo App

Once the models are trained, launch the interactive web dashboard to test custom inputs and visualize the attention weights live.

```bash
streamlit run demo_app/app.py
```
## 📊 Model Details

### 🔹 Baseline Model

**Architecture:**  
Unidirectional GRU Encoder + Unidirectional GRU Decoder

**Context Passing:**  
Only the **final hidden state** of the encoder is passed to the decoder as a fixed-length context vector.

**Limitation:**  
- Suffers from an **information bottleneck**
- Performance degrades on **longer sequences**
- Earlier input information may be lost

---

### 🔹 Attention-Based Model

**Architecture:**  
Bidirectional GRU Encoder + Unidirectional GRU Decoder

**Attention Mechanism:**  
Bahdanau (Additive) Attention

**Context Handling:**  
- The decoder computes a **dynamic context vector** at every decoding step
- Attention weights allow the decoder to focus on **relevant encoder hidden states**

**Advantages:**  
- Eliminates the fixed-size context bottleneck  
- Effectively handles **longer sequences**
- Learns a **clear diagonal alignment** in attention maps for the string reversal task

---

### 🔍 Comparison Summary

| Aspect               | Baseline Model              | Attention Model                     |
|---------------------|-----------------------------|-------------------------------------|
| Encoder             | Unidirectional GRU          | Bidirectional GRU                   |
| Context Vector      | Fixed (final encoder state) | Dynamic (attention-weighted)        |
| Bottleneck Issue    | Yes                         | No                                  |
| Long Sequence Handling | Weak                    | Strong                              |
| Alignment Learning  | None                        | Clear diagonal alignment            |
