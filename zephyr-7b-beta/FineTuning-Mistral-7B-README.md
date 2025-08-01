
---

# Fine-Tuning Mistral-7B on an Instruction-Based Dataset

This project focuses on fine-tuning the Mistral-7B model, specifically the `HuggingFaceH4/zephyr-7b-beta` variant, using an instruction-based dataset. The primary objective is to improve the model's ability to follow instructions accurately and enhance its performance on downstream tasks. By leveraging a symbolic instruction dataset, we aim to refine the model's instruction-following capabilities.

---

## 🧠 Model Selection

- **Model**: `HuggingFaceH4/zephyr-7b-beta`
- **Description**: A large, open-source, instruction-tuned model optimized for efficiency and performance in instruction-following tasks. Its architecture and pre-training make it an excellent candidate for fine-tuning on specialized instruction datasets.

---

## 📊 Dataset Preparation

- **Dataset**: `tatsu-lab/alpaca`
- **Formatting**: Each instance is structured as a chat-style instruction, prefixed with a consistent system prompt:  
  _"You are a helpful assistant. Please follow the instruction and provide a helpful response."_
- **Template**: A custom chat template was applied to standardize the instructions, ensuring compatibility with the model's input requirements.

---

## ⚙️ Training Configuration

- **Method**: Quantized Low-Rank Adaptation (QLoRA)
- **Frameworks**: 
  - Hugging Face Transformers
  - PEFT (Parameter-Efficient Fine-Tuning)
  - TRL (Transformers Reinforcement Learning)
- **Trainer**: `SFTTrainer` from the TRL library
- **Training Parameters**:
  - **Epochs**: 1
  - **Batch Size**: 4
  - **Gradient Accumulation Steps**: 2
  - **Learning Rate**: 2e-4
  - **Optimizer**: `paged_adamw_32bit`
  - **Precision**: Mixed precision with `fp16`
- **LoRA Configuration**:
  - **Rank (`r`)**: 8
  - **Alpha**: 32
  - **Dropout**: 0.05
  - **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`

---

## 📈 Evaluation

The model's performance was evaluated after fine-tuning using a fixed set of prompts. The following metrics were computed to assess improvements in instruction-following capabilities:

- **F1-Score**
- **ROUGE-L**
- **BERTScore**

**Note**: These metrics were specifically calculated post-fine-tuning to quantify the enhancements achieved. Pre-fine-tuning evaluation was conducted but did not include these specific metrics.

---

