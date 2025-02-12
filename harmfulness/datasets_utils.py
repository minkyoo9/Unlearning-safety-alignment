# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number

from constants import PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER

TASK_PROMPT_FORMAT = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: '''

# def format_prompt(
#     input: str | list[str],  # pylint: disable=redefined-builtin
#     eos_token: str,
# ) -> str:
#     if isinstance(input, str):
#         input = [input]
#     elif not isinstance(input, list):
#         raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str or list[str].')

#     if len(input) % 2 != 1:
#         raise ValueError(
#             'The length of `input` must be odd, while `input` must end at the user question.',
#         )

#     buffer = [PROMPT_BEGIN]
#     for i, line in enumerate(input):
#         if i % 2 == 0:
#             # User input
#             buffer.extend((PROMPT_USER.format(input=line), PROMPT_ASSISTANT))
#         else:
#             # Assistant response
#             buffer.extend((line, eos_token))

#     return ''.join(buffer)

def format_prompt(
    input: str,
    tokenizer,
    task_prompt=False
) -> str:

    if task_prompt:
        return TASK_PROMPT_FORMAT.format(instruction=input)
    else:
        return tokenizer.apply_chat_template([{"role": "user", "content": input}], tokenize=False, add_generation_prompt=True)

def format_prompt_fewshot(
    input: str,
    tokenizer,
    fewshot_examples,
    direct_answer_trigger,
) -> str:

    prompt = []
    for fewshot_example in fewshot_examples:
        question = fewshot_example['question']
        answer = fewshot_example['answer']
        final_answer = fewshot_example['final_answer']

        prompt.append({"role": "user", "content": f'{question}'})
        # prompt.append({"role": "assistant", "content": f'Let’s think step by step. {answer} \n{direct_answer_trigger}{final_answer}'})
        prompt.append({"role": "assistant", "content": f'{answer} \n{direct_answer_trigger}{final_answer}'})

    prompt.append({"role": "user", "content": input})

    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

def right_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def left_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return right_padding(
        [seq.flip(0) for seq in sequences],
        padding_value=padding_value,
    ).flip(1)
