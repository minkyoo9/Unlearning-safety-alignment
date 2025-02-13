# Unlearning-Safety-Alignment

Implementation of *Refusal Is Not an Option: Unlearning Safety Alignment of LLMs*, under review as a conference paper at **USENIX Security 2025**.

**Note:** This implementation is based on the following repositories:
- [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf/tree/main)
- [RWKU](https://github.com/jinzhuoran/RWKU)
- [MUSE](https://github.com/swj0419/muse_bench)

---

## Installation and Setup

### **1. Install Anaconda**
Download and install Anaconda from [here](https://www.anaconda.com/download).

### **2. Create and Activate Python Environment**
```bash
conda env create -f environment.yaml
conda activate unlearning
```

### **3. Set Your Hugging Face Token**
1. Insert your Hugging Face token in the following locations (used for training):
   - `hf_hub_token` in `LLaMA-Factory/src/llmtuner/hparams/model_args.py`
   - `token` in `muse_bench/utils.py`
2. Set the environment variable for Hugging Face token (used for evaluation):
```bash
export HF_TOKEN=<hf_api_key>
```

---

## Scenario I: Unlearning Safety Alignment

### **Step 1: Construct Unlearning Dataset (Rejection Responses)**
#### **1. Extract Model Responses Against the AdvBench Dataset**
And use LLaMA-Guard to classify responses based on their safety:
```bash
cd harmfulness
./run_eval_harmful.sh
```
**Output:** Results are stored in `outs_llama/out_AdvBench`

#### **2. Process Rejection Responses**
Extract only safe responses to create the unlearning dataset:
```bash
python rejection_responses_processing.py
```
**Output:** The processed dataset is stored in `data/unlearning_data_AdvBench_3B.json`

---

### **Step 2: Conduct Knowledge Unlearning**
**Note:** All hyperparameters are set for training on two **A100 80GB** GPUs.

#### **1. Using DPO, NPO, or GA**
Navigate to the LLaMA-Factory scripts directory and run the appropriate method:
```bash
cd LLaMA-Factory/scripts/full
./run_{method}.sh
```
**Output:**
- Model and training results are saved in `saves_3B/{method}_full_adv/${lr}`
- Evaluation results are stored in `results_3B/{method}_full_adv/${lr}`

#### **2. Using TV**
Run the unlearning process using the TV method:
```bash
cd muse_bench/baselines/scripts
./run_tv.sh
```
**Output:** The trained model is stored in `tv`

---

### **Step 3: Evaluate Harmfulness Post-Unlearning**
To evaluate the safety of the unlearned model, run:
```bash
cd harmfulness
./run_eval_harmful.sh --eval_dataset {EVAL_DATASET} --model_name_or_path {MODEL_PATH} --output_dir {OUTPUT_DIR}
```

**Example:**
```bash
./run_eval_harmful.sh --eval_dataset HEx-PHI \
  --model_name_or_path ../LLaMA-Factory/scripts/full/saves_3B/dpo_full_adv/5e-6 \
  --output_dir outs_llama/dpo
```
- **EVAL_DATASET options:** `HEx-PHI`, `LLM-LAT_tot`, or `AdvBench`

---

