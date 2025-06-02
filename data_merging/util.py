# util.py

import re
from rouge import Rouge

def extract_data_content(text):
    pattern = re.compile(r'\[DATA\](.*?)\[/DATA\]', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return "No data found"

def data_parsing(replies):
    parsed_data = []
    for d in replies:
        parsed_data.append({'text': extract_data_content(d)})
    return parsed_data

def initial_prompt(current_data, content=None, prompt_initial=None):
    prompts_initial = []
    if content is None:
        for s in current_data:
            p_ = prompt_initial
            p_ = p_.replace('[ORIGINAL]', s['text'])
            prompts_initial.append(p_)
    else:
        for s, c in zip(current_data, content):
            p_ = prompt_initial
            p_ = p_.replace('[ORIGINAL]', s['text'])
            p_ = p_.replace('[CONTENT]', c)
            prompts_initial.append(p_)
    return prompts_initial

def rewrite_prompt(original, content=None, prompt_rewrite=None):
    prompts_rewrite = []
    if content is None:
        for o in original:
            p_ = prompt_rewrite
            p_ = p_.replace('[ORIGINAL]', o['text'])
            prompts_rewrite.append(p_)
    else:
        for o, c in zip(original, content):
            p_ = prompt_rewrite
            p_ = p_.replace('[ORIGINAL]', o['text'])
            p_ = p_.replace('[CONTENT]', c)
            prompts_rewrite.append(p_)
    return prompts_rewrite

def calculate_similarity(list1, list2=None, list3=None):
    rouge = Rouge()
    texts1 = [item['text'] for item in list1]
    if list2 is not None and list3 is not None:
        texts2 = [item for item in list2]
        texts3 = [item['text'] for item in list3]
        rouge_scores_list1_list2 = []
        rouge_scores_list1_list3 = []
        for text1, text2 in zip(texts1, texts2):
            score = rouge.get_scores(text1, text2)[0]['rouge-l']['r']
            rouge_scores_list1_list2.append(score)
        for text1, text3 in zip(texts1, texts3):
            score = rouge.get_scores(text1, text3)[0]['rouge-l']['r']
            rouge_scores_list1_list3.append(score)
        return rouge_scores_list1_list2, rouge_scores_list1_list3
    elif list3 is not None:
        texts3 = [item['text'] for item in list3]
        rouge_scores_list1_list3 = []
        for text1, text3 in zip(texts1, texts3):
            score = rouge.get_scores(text1, text3)[0]['rouge-l']['r']
            rouge_scores_list1_list3.append(score)
        return rouge_scores_list1_list3
    else:
        raise ValueError("Invalid input for calculate_similarity")

def extract_eval_scores(replies_eval):
    semantic_scores, syntactic_scores = [], []
    semantic_pattern = re.compile(r'\[SEMANTIC\]\s*(\d+\.\d+)')
    syntactic_pattern = re.compile(r'\[SYNTACTIC\]\s*(\d+\.\d+)')
    for ind, reply in enumerate(replies_eval):
        semantic_match = semantic_pattern.search(reply)
        if semantic_match:
            semantic_scores.append(float(semantic_match.group(1)))
        else:
            semantic_scores.append(None)
        syntactic_match = syntactic_pattern.search(reply)
        if syntactic_match:
            syntactic_scores.append(float(syntactic_match.group(1)))
        else:
            syntactic_scores.append(None)
    return semantic_scores, syntactic_scores

def run_initial(prompts_initial, prompt_initial_system, client):
    from tqdm import tqdm
    replies_initial = []
    for pt in tqdm(prompts_initial):
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_initial_system},
                {"role": "user", "content": pt},
            ],
            model="gpt-4o-2024-05-13",
        )
        replies_initial.append(chat_completion.choices[0].message.content)
    return replies_initial

def run_eval(prompts_eval, prompt_evaluate_system, client):
    from tqdm import tqdm
    replies_eval = []
    for pt in tqdm(prompts_eval):
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_evaluate_system},
                {"role": "user", "content": pt},
            ],
            model="gpt-4o-2024-05-13",
        )
        replies_eval.append(chat_completion.choices[0].message.content)
    return replies_eval

def run_rewrite(prompts_rewrite, results_dict, num, prompt_rewrite_system, prompt_rewrite2, client):
    from tqdm import tqdm
    replies_rewrite = []
    for ind, pt in tqdm(enumerate(prompts_rewrite), total=len(prompts_rewrite)):
        for i in range(num-1):
            results = results_dict[i+1]
            data, s, s_ori, s_con, s_sem, s_syn = results[ind]
            pt += f"Synthetic Data_{i+1}: {data['text']}\n S:{s}\n S_ori:{s_ori}\n S_con:{s_con}\n S_sem:{s_sem}\n S_syn:{s_syn}\n"
        pt += prompt_rewrite2
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_rewrite_system},
                {"role": "user", "content": pt},
            ],
            model="gpt-4o-2024-05-13",
        )
        replies_rewrite.append(chat_completion.choices[0].message.content)
    return replies_rewrite

def rewrite_process(
    reject_data, content, results_dict, iter_,
    prompt_initial, prompt_initial_system, prompt_rewrite, prompt_rewrite_system, prompt_rewrite2,
    client
):
    if iter_ == 1:
        prompts_initial = initial_prompt(reject_data, content, prompt_initial=prompt_initial)
        replies_initial = run_initial(prompts_initial, prompt_initial_system, client)
        replies_current = data_parsing(replies_initial)
    else:
        prompts_rewrite = rewrite_prompt(reject_data, content, prompt_rewrite=prompt_rewrite)
        replies_rewrite = run_rewrite(prompts_rewrite, results_dict, iter_, prompt_rewrite_system, prompt_rewrite2, client)
        replies_current = data_parsing(replies_rewrite)
    return replies_current

def eval_prompt(initial_rewritten_data, prompt_evaluate=None):
    prompts_eval = []
    for s in initial_rewritten_data:
        p_ = prompt_evaluate
        p_ = p_.replace('[SYNTHETIC]', s['text'])
        prompts_eval.append(p_)
    return prompts_eval

def eval_process(
    replies, reject_data, content, results_dict, iter_,
    prompt_evaluate, prompt_evaluate_system, client
):
    prompts_eval = eval_prompt(replies, prompt_evaluate)
    replies_eval = run_eval(prompts_eval, prompt_evaluate_system, client)
    semantic_scores, syntactic_scores = extract_eval_scores(replies_eval)
    if content is not None:
        sim_con, sim_ori = calculate_similarity(replies, list2=content, list3=reject_data)
    else:
        sim_con, sim_ori = None, calculate_similarity(replies, list3=reject_data)
    for i, (data, s_ori, s_con, s_sem, s_syn) in enumerate(zip(replies, (sim_ori if sim_ori is not None else [None]*len(replies)),
                                                               (sim_con if sim_con is not None else [None]*len(replies)),
                                                               semantic_scores, syntactic_scores)):
        s = 2*s_ori + (s_con if s_con is not None else 0) + 0.5*((s_sem or 0)+(s_syn or 0))
        results_dict[iter_].append((data, s, s_ori, s_con, s_sem, s_syn))
    return results_dict
