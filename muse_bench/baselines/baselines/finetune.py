from .dataset import DefaultDataset
from .utils import load_model_and_tokenizer

import transformers
from torch.utils.data import DataLoader
from pdb import set_trace

def finetune(
    model_dir: str,
    data_file: str,
    out_dir: str,
    epochs: int = 5,
    per_device_batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None
):
    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    dataset = DefaultDataset( 
        data_file,
        tokenizer=tokenizer,
        max_len=max_len
    ) 
    

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='no',
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        do_train=True,
        gradient_accumulation_steps=2,
        report_to='none'        # Disable wandb
    )
    # print(training_args)
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn()
    )

    model.config.use_cache = False  # silence the warnings.
    print('Train start')
    trainer.train()
    trainer.save_model(out_dir)
