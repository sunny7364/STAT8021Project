from datasets import Dataset, Features, Value
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
import json
import os
import random
from tqdm import tqdm
import sys

from ipdb import set_trace

output_file = "output.txt"
sys.stdout = open(output_file, "a")
os.environ['OPENAI_API_KEY'] = "YOUR API KEY"

matrix_names = [faithfulness, answer_relevancy]
input_file = 'movie_result.json'

data_list = []
with open(input_file) as f:
    for line in f:
        # if line.strip():
        #     lines.append(line)
        #     if line.strip() == '}':
        data = json.loads(line)
        if data['Answer'] == "error":
            continue
        data_list.append(data)

print(input_file)

data_length = len(data_list)
random_indexes = random.sample(range(data_length), 25)
grouped_random_indexes = [random_indexes[i:i+5] for i in range(0, len(random_indexes), 5)]

questions = [data['Question'] for data in data_list]
answers = [data['Answer'] for data in data_list]
contexts = [[data['System Prompt']] for data in data_list]

for idx_group in tqdm(grouped_random_indexes):
    questions_ = [questions[i] for i in idx_group]
    answers_ = [answers[i] for i in idx_group]
    contexts_ = [contexts[i] for i in idx_group]

    merged_data = {
        'question': questions_,
        'answer': answers_,
        'contexts': contexts_
    }

    # print(len(merged_data["question"]))
    # print(len(merged_data["answer"]))
    # print(len(merged_data["contexts"]))
    print(idx_group)

    # breakpoint()

    for i, matrix in enumerate(matrix_names):
        dataset = Dataset.from_dict(merged_data)
        score = evaluate(dataset,metrics=[matrix])
        score.to_pandas()
        print("-"*20 + f"{i} {matrix.name}: {score}" + "-"*20)

sys.stdout.close()
sys.stdout = sys.__stdout__


import statistics

with open('output2.txt', 'r') as file:
    lines = file.readlines()

faithfulness_list = []
answer_relevancy_list = []

for line in lines:
    if 'faithfulness' in line:
        faithfulness_str = line.split(': ')[2].strip('}\n')
        faithfulness = float(faithfulness_str.split('}')[0])
        faithfulness_list.append(faithfulness)
    elif 'answer_relevancy' in line:
        answer_relevancy_str = line.split(': ')[2].strip('}\n')
        answer_relevancy = float(answer_relevancy_str.split('}')[0])
        answer_relevancy_list.append(answer_relevancy)

datas = [faithfulness_list, answer_relevancy_list]

for data in datas:
    mean = statistics.mean(data)
    std = statistics.stdev(data)

    print("Mean:", mean)
    print("Standard Deviation:", std)