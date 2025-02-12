# Inspired by: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

import math
import os.path
import json
from typing import TYPE_CHECKING, List, Optional
import torch
from transformers import AutoTokenizer
from ...data import PairwiseDataCollatorWithPadding, get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer


from ...eval import *
from datasets import Dataset
from pdb import set_trace


FORGET_LEVEL1 = 'forget_level1.json'
FORGET_LEVEL2 = 'forget_level2.json'
FORGET_LEVEL3 = 'forget_level3.json'
NEIGHBOR_LEVEL1 = 'neighbor_level1.json'
NEIGHBOR_LEVEL2 = 'neighbor_level2.json'

RETAIN_MMLU = 'retain_mmlu.json'
RETAIN_BBH = 'retain_bbh.json'
TRUTHFUL = 'truthful.json'
TRIVIAQA = 'triviaqa.json'
FLUENCY = 'fluency.json'
FORGET_MIA = 'forget_mia.json'
RETAIN_MIA = 'retain_mia.json'

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    tokenizer.pad_token = tokenizer.eos_token
    with open('../../../harmfulness/data/wiki_data_sampled.json', 'r') as file: #llama
        chosen_data = json.load(file)
    with open(finetuning_args.unlearn_data, 'r') as file:
        rejected_data = json.load(file)
    print(finetuning_args.unlearn_data)
    chosen_data = chosen_data*(len(rejected_data)//len(chosen_data)+1) ## for the case of chosen data is shorter than rejected data
    prompt_ids = []
    chosen_ids = []
    rejected_ids = []

    ## Create dataset
    for chosen_text, rejected_text in zip(chosen_data, rejected_data):
        prompt_ids.append([])
        chosen_encoded = tokenizer.encode(chosen_text['text'])
        rejected_encoded = tokenizer.encode(rejected_text['text'])
    
        chosen_ids.append(chosen_encoded)
        rejected_ids.append(rejected_encoded)
    
    dataset = Dataset.from_dict({
        'prompt_ids': prompt_ids,
        'chosen_ids': chosen_ids,
        'rejected_ids': rejected_ids
    })
        
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = PairwiseDataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Create reference model
    if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
        ref_model = model
    else:
        ref_model = create_ref_model(model_args, finetuning_args)

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        if model_args.save_model:
            trainer.save_model()
            trainer.save_state()
            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])


    eval_dataset_dir = data_args.eval_dataset_dir
    target = data_args.target
    eval_dataset_dir = os.path.join(eval_dataset_dir, target)

    with open(os.path.join(eval_dataset_dir, RETAIN_MMLU), 'r') as f:
        retain_mmlu = json.load(f)
    with open(os.path.join(eval_dataset_dir, RETAIN_BBH), 'r') as f:
        retain_bbh = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRUTHFUL), 'r') as f:
        truthfulqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, TRIVIAQA), 'r') as f:
        triviaqa = json.load(f)
    with open(os.path.join(eval_dataset_dir, FLUENCY), 'r') as f:
        fluency = json.load(f)


    output_result_dir = os.path.join(data_args.output_result_dir, target)
    os.makedirs(os.path.join(output_result_dir), exist_ok=True)

    model.eval()
    with torch.no_grad():
        e_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side='left')
        e_tokenizer.pad_token = e_tokenizer.eos_token
        # for llama setting
        print("Evaluate mmlu...")
        eval_mmlu(model, e_tokenizer, retain_mmlu, batch_size=2, output_result_dir=os.path.join(output_result_dir, 'mmlu.json'), use_prompt=data_args.use_prompt)
        print("Evaluate bbh...")
        eval_bbh(model, e_tokenizer, retain_bbh, batch_size=32, output_result_dir=os.path.join(output_result_dir, 'bbh.json'), use_prompt=data_args.use_prompt)
        print("Evaluate truthful...")
        eval_truthfulqa(model, e_tokenizer, truthfulqa, batch_size=8, output_result_dir=os.path.join(output_result_dir, 'truthful.json'), use_prompt=data_args.use_prompt)
        print("Evaluate triviaqa...")
        eval_triviaqa(model, e_tokenizer, triviaqa, batch_size=32, output_result_dir=os.path.join(output_result_dir, 'triviaqa.json'), use_prompt=data_args.use_prompt)
        print("Evaluate fluency...")
        eval_fluency(model, e_tokenizer, fluency, batch_size=32, output_result_dir=os.path.join(output_result_dir, 'fluency.json'), use_prompt=data_args.use_prompt)
