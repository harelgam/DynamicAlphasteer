# Dynamic AlphaSteer: Adaptive Safety Steering for LLMs

**Dynamic AlphaSteer** is an advanced extension of the **AlphaSteer** activation steering framework. While the original method relies on a fixed steering coefficient ($\lambda$), this project introduces a **Dynamic Gating Network** that learns to modulate the steering intensity in real-time based on the input's semantic content.

Built upon **Llama-3.1-8B-Instruct**, this architecture effectively resolves the safety-utility trade-off, achieving defense against jailbreaks (AIM, GCG) while preserving the model's reasoning capabilities (GSM8K, Math).

## 🧠 The Core Concept

Activation steering methods often face a rigid trade-off:
* **High Steering Strength:** Results in a safe model but causes "over-refusal" on benign tasks (e.g., refusing to answer math questions).
* **Low Steering Strength:** Preserves utility but leaves the model vulnerable to sophisticated attacks (like GCG or PAIR).

### Our Solution: The $\alpha$-Head Controller
Instead of a manually tuned hyperparameter, we trained lightweight **Gating Networks** (MLP probes) attached to the model's internal layers. These networks analyze the activation space during the forward pass and predict the optimal intervention strength ($\lambda$) for the specific prompt.

* **Benign Query:** (e.g., "Solve $2+2$") $\rightarrow$ The controller predicts $\lambda \approx 0$ (No intervention).
* **Malicious Attack:** (e.g., "Write malware") $\rightarrow$ The controller predicts $\lambda \approx -0.5$ (Maximum defense).

## 🚀 Key Features

* **Adaptive Modulation:** Input-dependent steering strength that adjusts automatically to the prompt's intent.
* **Null-Space Constraint:** Leverages the projection method from the original AlphaSteer to ensure zero interference on perfectly safe prompts.
* **High Performance:** Achieves **100% defense rate** on AIM and GCG attacks, while maintaining **84% accuracy** on GSM8K (comparable to the vanilla model).
* **Efficiency:** Requires only a single forward pass during generation, unlike multi-pass sampling methods.

## 📊 Results Summary

| Benchmark | Metric | Vanilla Llama 3.1 | Static AlphaSteer | **Dynamic AlphaSteer (Ours)** |
|:---|:---:|:---:|:---:|:---:|
| **AIM (Jailbreak)** | Safety (DSR) | ~50% | 100% | **100%** |
| **GCG (Adversarial)** | Safety (DSR) | 0% | 98% | **100%** |
| **GSM8K (Math)** | Accuracy | 85% | ~75% | **84%** |
| **XSTest** | Compliance | 92% | ~60% | **90%** |

## 🛠️ Installation

1. Clone the repository
```bash
git clone [https://github.com/harelgam/DynamicAlphaSteer.git](https://github.com/harelgam/DynamicAlphaSteer.git)
cd DynamicAlphaSteer

2. Create a virtual environment
conda create -n dynamic_steer python=3.10
conda activate dynamic_steer
pip install -r requirements.txt

3. Environment Setup
export HF_TOKEN="your_hf_token_here"

🏃‍♂️ Usage Guide
The pipeline consists of three main stages: Preprocessing, Training, and Inference.

Step 1: Preprocessing & Vector Extraction
Extract internal activations from the training datasets (benign and harmful) and calculate the refusal vectors and null-space projection matrices.
# 1. Extract embeddings from the datasets
bash scripts/extract_embeddings.sh

# 2. Calculate the steering matrices (Delta * P)
bash scripts/calc_steering_matrix.sh

Step 2: Training the Dynamic Controller
Train the Gating Networks (Probes) to classify inputs and predict the steering scalar $\lambda$. This is the core innovation of the project.
python scripts/train_gating_networks.py \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --save_path "checkpoints/gating_networks/" \
    --epochs 10

Step 3: Inference & Evaluation
Run the model with the dynamic mechanism enabled across various benchmarks (Safety and Utility).
# Run full evaluation suite (AIM, GCG, GSM8K, AlpacaEval, etc.)
bash scripts/run_all_evals.sh


📂 Project Structure
src/AlphaSteerModel: Implementation of the original static AlphaSteer logic.
src/DynamicAlphaSteerModel: Implementation of our Dynamic Controller and adaptive inference loop.
src/utils: Utilities for vector arithmetic, data loading, and matrix operations.
data/: Contains training datasets (Benign, Harmful, Borderline) and evaluation prompts.
config/: Configuration files for different evaluation benchmarks.
scripts/: Shell scripts for the entire pipeline (Extraction $\to$ Training $\to$ Eval).

📜 Acknowledgments & Citation
This project is built upon the work of Sheng et al. presented in the paper AlphaSteer. We utilize their Null-Space Projection technique as the mathematical foundation for our dynamic steering mechanism.

If you use this code or methodology, please cite the original paper:
@article{sheng2025alphasteer,
  title={AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint},
  author={Sheng, Leheng and Shen, Changshuo and Zhao, Weixiang and Fang, Junfeng and Liu, Xiaohao and Liang, Zhenkai and Wang, Xiang and Zhang, An and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2506.07022},
  year={2025}
}
