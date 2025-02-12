# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
import os.path
import json
from typing import TYPE_CHECKING, List, Optional
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from ...data import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .trainer import CustomTrainer
from ..utils import create_modelcard_and_push, create_ref_model

from ...eval import *
from pdb import set_trace
from datasets import Dataset

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

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_npo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    tokenizer.pad_token = tokenizer.eos_token
    with open(finetuning_args.unlearn_data, 'r') as file:
        data = json.load(file)
    print(finetuning_args.unlearn_data)

    
    # Extract texts for tokenization
    texts = [item['text'] for item in data]
    
    # Tokenize data
    tokenized_data = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    
    # Convert to datasets.Dataset
    dataset = Dataset.from_dict({
        'input_ids': tokenized_data['input_ids'],
        'attention_mask': tokenized_data['attention_mask']
    })

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    if model_args.train_layers is not None:
        train_layers = model_args.train_layers.split('-')
        for name, param in model.named_parameters():
            if any(f'layers.{i}.' in name for i in range(int(train_layers[0]), int(train_layers[-1]))):
                param.requires_grad = True
                print('Trainable Module:', name)
            else:
                param.requires_grad = False

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    ref_model = create_ref_model(model_args, finetuning_args)
    # Initialize our Trainer
    trainer = CustomTrainer(
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
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])


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
