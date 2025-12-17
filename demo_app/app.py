import streamlit as st
import json
import os
import sys
import torch
import pickle
from io import BytesIO

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo_app.load_models import load_models
from demo_app.demo_utils import preprocess_input, postprocess_output, generate_attention_heatmap, get_reversed_string, get_vocab

# --- Configuration ---
SAMPLE_INPUTS_PATH = "demo_app/sample_inputs.json"
ATTENTION_SAMPLES_PATH = "models/attention/attention_samples.pkl" # Path to training logs

# --- Load Models and Vocab ---
@st.cache_resource
def load_all_resources():
    """Loads all models and vocab, caching the result."""
    return load_models()

vocab, baseline_model, attention_model = load_all_resources()

if vocab is None:
    st.error("Failed to load vocabulary. Please ensure data/char_vocab.json exists.")
    st.stop()

# --- Load Sample Inputs ---
try:
    with open(SAMPLE_INPUTS_PATH, 'r') as f:
        sample_inputs = json.load(f)
except FileNotFoundError:
    sample_inputs = ["hello", "attention", "reverse"]

# --- Streamlit App ---
st.set_page_config(
    page_title="Seq2Seq Attention Mechanism Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Seq2Seq Attention Mechanism: A Visual Tutorial")
st.markdown("""
This interactive demo compares a **Baseline Seq2Seq Model** (without Attention) 
and an **Attention-based Seq2Seq Model** on a simple character-level string reversal task. 
The goal is to visually demonstrate how the Attention Mechanism solves the information 
bottleneck problem of traditional encoder-decoder architectures.
""")

# --- Sidebar for Input ---
st.sidebar.header("Input String")

# Input selection
input_option = st.sidebar.selectbox(
    "Choose a sample input:",
    ["Custom Input"] + sample_inputs
)

if input_option == "Custom Input":
    user_input = st.sidebar.text_input(
        "Enter a string to reverse:",
        value="university"
    )
else:
    user_input = input_option

# Clean and validate input
input_string = user_input.strip().lower()
if not input_string:
    st.warning("Please enter a non-empty string.")
    st.stop()

# --- Model Inference ---
input_tensor, input_lengths = preprocess_input(input_string)
expected_output = get_reversed_string(input_string)

# Baseline Prediction
if baseline_model:
    baseline_predicted_indices = baseline_model.predict(input_tensor, input_lengths, vocab, max_length=len(input_string) + 1)
    baseline_output = postprocess_output(baseline_predicted_indices)
else:
    baseline_output = "Model not loaded."

# Attention Prediction
attention_output = "Model not loaded."
attention_heatmap_buf = None
if attention_model:
    attention_predicted_indices, attention_matrices = attention_model.predict(input_tensor, input_lengths, vocab, max_length=len(input_string) + 1)
    attention_output = postprocess_output(attention_predicted_indices)
    
    if attention_matrices is not None and attention_matrices.shape[0] > 0:
        # attention_matrices is (B, T_out, T_in), we take the first (and only) batch item
        attention_heatmap_buf = generate_attention_heatmap(attention_matrices[0], input_string, attention_output)

# --- Results Display ---
st.subheader("Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Input String**")
    # Using st.code prevents truncation and formatting issues common with st.metric
    st.code(input_string, language=None)

with col2:
    st.markdown("**Expected Output**")
    st.code(expected_output, language=None)

with col3:
    # Check if attention model is correct
    is_correct = "✅ Correct" if attention_output == expected_output else "❌ Incorrect"
    st.metric("Attention Model Status", is_correct)

st.markdown("---")

# --- Comparison Table ---
st.subheader("Model Comparison")
comparison_data = {
    "Model": ["Baseline (No Attention)", "Attention-based Seq2Seq"],
    "Predicted Output": [baseline_output, attention_output],
    "Correct?": ["✅ Correct" if baseline_output == expected_output else "❌ Incorrect", is_correct]
}
st.table(comparison_data)

# # --- Attention Visualization (Live) ---
# st.subheader("Live Attention Map (Final Model)")
# st.markdown("""
# The heatmap below visualizes the **Attention Weights** for the input you provided above. 
# You should see a strong **anti-diagonal** pattern (bottom-left to top-right) if the model is working correctly.
# """)

# if attention_heatmap_buf:
#     st.image(attention_heatmap_buf, caption=f"Attention Weights for '{input_string}'", use_container_width=True)
# else:
#     st.info("Attention map could not be generated. Check model loading and prediction logic.")

st.markdown("---")

# --- Training Progression Section (New Feature) ---
st.subheader("Visualizing Training Progress")
st.markdown("""
How did the model learn this alignment? Below are snapshots of the attention matrix taken during training 
on a fixed validation sample. You can see the pattern evolve from **random noise** to a **clear diagonal**.
""")

# Load saved training samples
@st.cache_data
def load_training_samples():
    if os.path.exists(ATTENTION_SAMPLES_PATH):
        try:
            with open(ATTENTION_SAMPLES_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading training samples: {e}")
            return None
    return None

training_data = load_training_samples()

if training_data:
    sample_input = training_data['input']
    sample_target = training_data['target']
    epochs_data = training_data['epochs']
    
    # We want Epoch 1, Epoch 8, and the Final Epoch
    # Find these specific epochs in the list
    epoch_1 = next((x for x in epochs_data if x['epoch'] == 1), None)
    epoch_8 = next((x for x in epochs_data if x['epoch'] == 8), None)
    epoch_final = epochs_data[-1] if epochs_data else None

    # Create columns for the display
    t_col1, t_col2, t_col3 = st.columns(3)
    
    # Helper to display heatmap in a column
    def display_training_heatmap(column, epoch_data, title):
        if epoch_data:
            with column:
                st.markdown(f"**{title}**")
                # Generate plot using the utility
                buf = generate_attention_heatmap(
                    epoch_data['attention_matrix'], 
                    sample_input, 
                    sample_target
                )
                if buf:
                    st.image(buf, use_container_width=True)
                else:
                    st.write("Error generating plot.")
        else:
            column.warning(f"{title} data not found.")

    display_training_heatmap(t_col1, epoch_1, "Epoch 1 (Beginning)")
    display_training_heatmap(t_col2, epoch_8, "Epoch 8 (Middle)")
    display_training_heatmap(t_col3, epoch_final, f"Epoch {epoch_final['epoch']} (Final)")
    
else:
    st.warning(f"Training log file not found at {ATTENTION_SAMPLES_PATH}. Run 'train_attention.py' to generate training history.")

st.markdown("---")
st.caption("Project by Amila Vidana Gamage. Models trained on character-level string reversal.")