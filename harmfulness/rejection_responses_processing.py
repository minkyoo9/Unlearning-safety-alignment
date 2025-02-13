import json
with open("outs_llama/out_AdvBench/predictions.json", "r") as file:
    predictions = json.load(file)
safe_answers = []
for data in predictions:
    if data['safety'] == 'safe':
        safe_answers.append(data['response'])
safe_answers_fin = []
for d in safe_answers:
    safe_answers_fin.append({'text':d})
print(f'Total number of safe responses are {len(safe_answers_fin)}')
with open('data/unlearning_data_AdvBench_3B.json', 'w') as file:
    json.dump(safe_answers_fin, file)