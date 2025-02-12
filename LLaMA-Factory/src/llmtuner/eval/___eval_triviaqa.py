import torch
import numpy as np
from .inference import generate_completions
import json
import os
import re
import string
from collections import Counter
from pdb import set_trace

class Metric:
    def __init__(self, name, **kwargs):
        self.name = name
        self.store_individual_scores = False

    def __call__(self, predictions, references, questions=None, ids=None):
        raise NotImplementedError()

    @classmethod
    def _normalize_text(cls, text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = text.lower()
        text = "".join(char for char in text if char not in set(string.punctuation))
        text = re.sub(regex, " ", text)
        text = " ".join(text.split())
        return text

    def _get_tokens(self, text):
        if not text:
            return []
        return self._normalize_text(text).split()

class F1(Metric):
    """Computes average F1 score between a list of predictions and a list of
    list of references.

    Code taken from: https://github.com/McGill-NLP/topiocqa
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._f1(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]
        return {"f1": np.mean(scores)}

    def _f1(self, prediction, references):
        """Computes F1 score between a prediction and a list of references.
        Take the max F1 score if there are multiple references.
        """

        f1_scores = [self._f1_score(prediction, reference) for reference in references]
        return max(f1_scores)

    def _f1_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0 or len(prediction_tokens) == 0:
            # If either is empty, then F1 is 1 if they agree, 0 otherwise.
            return int(reference_tokens == prediction_tokens)

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)
        recall = 1.0 * num_common / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1


class EM(Metric):
    """Computes average exact match score between a list of predictions and a
    list of list of references.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._exact_match(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]
        return {"em": np.mean(scores)}

    def _exact_match(self, prediction, references):
        """Computes exact match score between a prediction and a list of
        references. Take the max EM score if there are multiple references.
        """

        em_scores = [
            self._exact_match_score(prediction, reference) for reference in references
        ]
        return max(em_scores)

    def _exact_match_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        return int(reference_tokens == prediction_tokens)

few_prompt_list = [
 "Q: Who was President when the first Peanuts cartoon was published?\n",
 "A: Harry Truman\n",
 "Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\n",
 "A: Sinclair Lewis\n",
 "Q: Where in England was Dame Judi Dench born?\n",
 "A: York\n",
 "Q: William Christensen of Madison, New Jersey, has claimed to have the world's biggest collection of what?\n",
 "A: Beer Cans\n",
 "Q: In which decade did Billboard magazine first publish and American hit chart?\n",
 "A: 30s\n"]


@torch.no_grad()
def eval_triviaqa(model, tokenizer, dataset, batch_size=1, output_result_dir=None, use_prompt=False):
    tokenizer.padding_side = 'left'
    prompts = []
    questions = []
    answers = []

    for sample in dataset:
        question = sample['question']
        prompt = f'Following previous examples, breifly answer the following question. \n Q: {question}\n'
        messages = []
        for ind, d in enumerate(few_prompt_list):
            if ind%2 == 0: # question
                messages.append({"role": "user", "content": d})
            else: # answer
                messages.append({"role": "assistant", "content": d})
        messages.append({"role": "user", "content": prompt})
        prompt = tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
        prompt += "A:"
        # set_trace()
        prompts.append(prompt)
        answers.append(sample['answers'])
        questions.append(sample)

    terminators = [
        [tokenizer.eos_token_id],
        [tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    ]
    # set_trace()
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=30,
        batch_size=batch_size,
        do_sample=False,
        stop_id_sequences=terminators
    )
    for sample, output in zip(dataset, outputs):
        sample['prediction'] = output
    em = EM('em')
    f1 = F1('F1')
    em_score = em(outputs, answers)
    f1_score = f1(outputs, answers)
    # set_trace()
    print("EM {:.3f}".format(em_score['em']))
    print("F1 {:.3f}".format(f1_score['f1']))
    output_result = {
        'EM': em_score['em'],
        'F1': f1_score['f1'],
        'results': dataset,
    }

    if output_result_dir is not None:
        with open(output_result_dir, 'w') as f:
            json.dump(output_result, f, indent=4)

    tokenizer.padding_side = 'right'

    return em_score['em'], f1_score['f1']

