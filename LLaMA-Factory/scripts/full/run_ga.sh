#!/bin/bash


names=('1_Stephen_King')

mkdir -p ./logs_3B/ga_full_adv

lrs=(7e-7)

for name in "${names[@]}"
do
    id=$name

    for lr in "${lrs[@]}"
    do
        echo ${lr}
        PYTHONPATH=./ WANDB_DISABLED=true python ../../src/train_bash.py --stage ga \
        --unlearn_data ../../../harmfulness/data/unlearning_data_AdvBench_3B.json \
        --model_name_or_path meta-llama/Llama-3.2-3B-Instruct --do_train --save_model \
        --dataset ${id}_Positive --dataset_dir ../../data --finetuning_type full \
        --output_dir ./saves_3B/ga_full_adv/${lr} --overwrite_cache \
        --overwrite_output_dir --cutoff_len 1024 --preprocessing_num_workers 16 \
        --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --gradient_accumulation_steps 2 \
        --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 --save_strategy no \
        --evaluation_strategy no --template llama3 \
        --learning_rate ${lr} --num_train_epochs 3 --val_size 0.0000001 --plot_loss \
        --output_result_dir ./results_3B/ga_full_adv/${lr}\
        --bf16 --eval_dataset_dir ../../data/RWKU/Target/ \
        --target ${id} 2>&1 | tee ./logs_3B/ga_full_adv/${lr}_log.txt
    done
done

