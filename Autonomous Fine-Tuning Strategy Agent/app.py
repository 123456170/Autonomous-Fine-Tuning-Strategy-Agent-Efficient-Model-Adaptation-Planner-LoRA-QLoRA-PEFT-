import streamlit as st

st.set_page_config(page_title="Autonomous Fine-Tuning Strategy Agent", layout="wide")

st.title("🤖 Autonomous Fine-Tuning Strategy Agent")
st.subheader("Efficient Model Adaptation Planner (LoRA / QLoRA / PEFT)")

st.markdown(
    """
This tool generates a **production-ready fine-tuning strategy** for any AI task using
modern PEFT methods like **LoRA and QLoRA**.
"""
)

# -------------------------
# INPUTS
# -------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    base_model = st.text_input("Base Model", "meta-llama/Llama-2-7b-hf")
    task_type = st.selectbox(
        "Task Type",
        ["Text Classification", "Chat Assistant", "Summarization", "Code Generation", "Domain QA"]
    )

    dataset_size = st.selectbox(
        "Dataset Size",
        ["<1K", "1K-10K", "10K-100K", "100K+"]
    )

    compute = st.selectbox(
        "Compute Level",
        ["Low (1 GPU)", "Medium (2-4 GPUs)", "High (8+ GPUs)"]
    )

    domain = st.text_input("Domain", "general")

# -------------------------
# STRATEGY ENGINE
# -------------------------
def generate_strategy(task_type, dataset_size, compute, domain):
    # DATA STRATEGY
    data_strategy = f"""
### 1. 📊 Data Strategy
- Domain: {domain}
- Task: {task_type}

**Cleaning**
- Remove duplicates, PII, malformed samples
- Normalize formatting (especially for chat/code tasks)

**Formatting**
- Use instruction format:
  - Input → Instruction → Output structure
- Convert dataset into JSONL for HF Trainer compatibility

**Augmentation**
- Paraphrasing for small datasets
- Back-translation for QA/summarization
- Code mutation (for code tasks)
"""

    # PEFT STRATEGY
    lora_r = 8 if dataset_size == "<1K" else 16 if dataset_size == "1K-10K" else 32
    alpha = lora_r * 2

    target_modules = (
        "q_proj, v_proj"
        if task_type in ["Chat Assistant", "Domain QA"]
        else "q_proj, k_proj, v_proj, o_proj"
    )

    peft_strategy = f"""
### 2. 🧠 Parameter-Efficient Fine-Tuning (LoRA/QLoRA)

**Recommended Setup**
- Method: LoRA (QLoRA if low compute)
- Rank (r): {lora_r}
- Alpha: {alpha}
- Dropout: 0.05

**Target Modules**
- {target_modules}

**Why this config?**
- Lower dataset size → smaller rank avoids overfitting
- Attention-only tuning improves stability + memory efficiency
"""

    # TRAINING RECIPE
    lr = 2e-4 if compute == "Low (1 GPU)" else 1e-4

    training = f"""
### 3. 🏋️ Training Recipe

- Optimizer: AdamW
- Learning Rate: {lr}
- Scheduler: Cosine decay with warmup (5%)
- Epochs:
  - <1K → 5-8 epochs
  - 1K-10K → 3-5 epochs
  - 10K+ → 2-3 epochs

- Batch Size:
  - Low compute: 2-8 (gradient accumulation recommended)
- Mixed Precision: fp16 / bf16
- Gradient Clipping: 1.0

**Stability Tips**
- Use gradient accumulation if OOM
- Enable checkpoint saving every epoch
"""

    # EVALUATION
    eval_plan = f"""
### 4. 📈 Evaluation Plan

**Offline Metrics**
- Loss convergence trend
- Task-specific metrics:
  - Classification → Accuracy / F1
  - QA → Exact Match / F1
  - Summarization → ROUGE
  - Code → Pass@k

**Validation Strategy**
- 80/20 split or K-Fold (small datasets)
- Track overfitting gap (train vs val loss)

**Stop Criteria**
- Validation loss plateaus for 2+ epochs
- No metric improvement for N steps
"""

    # DEPLOYMENT
    deployment = f"""
### 5. 🚀 Deployment Strategy

**Model Export**
- Merge LoRA adapters into base model OR keep adapters separate

**Serving Options**
- HuggingFace Transformers pipeline
- vLLM (for fast inference)
- Quantized deployment (4-bit / 8-bit)

**Optimization**
- Use torch.compile (if supported)
- KV cache enabled for chat systems
- Batch inference for throughput

**Production Tip**
- Store LoRA separately for fast updates without retraining full model
"""

    return data_strategy, peft_strategy, training, eval_plan, deployment


# -------------------------
# BUTTON
# -------------------------
if st.button("🚀 Generate Fine-Tuning Strategy"):
    data, peft, train, eval_p, deploy = generate_strategy(
        task_type, dataset_size, compute, domain
    )

    st.markdown(data)
    st.markdown(peft)
    st.markdown(train)
    st.markdown(eval_p)
    st.markdown(deploy)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("⚡ Built with Streamlit | PEFT Strategy Agent (No API Required)")