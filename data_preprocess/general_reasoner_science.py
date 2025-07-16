import re
import os
import datasets
from datasets import Dataset, concatenate_datasets
import json
import argparse

system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\nUser: Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='TIGER-Lab/WebInstruct-verified')
    parser.add_argument('--local-dir', default='~/webinstruct-verified')

    args = parser.parse_args()

    dataset = datasets.load_dataset(args.dataset_name)
    
    train_dataset = dataset['train'] 
    target_categories = {"Physics", "Chemistry", "Biology"}
    train_dataset = train_dataset.filter(lambda x: x.get("category") in target_categories)

	# TODO: use pervious math eval and gpqa as val
    test_dataset = dataset['test'].select(range(100)) 
    gpqa_dataset = datasets.load_dataset('ZhuofengLi/gpqa_mcq')['train']
    aime24_dataset = datasets.load_dataset('Maxwell-Jia/AIME_2024', split='train') # actually test set


    # add a row to each data item that represents a unique id
    def make_map_fn(data_source, split):

        def process_fn(example, idx):
            if 'question' in example:
                question_raw = example.pop('question')
            else:
                question_raw = example.pop('Problem')
            if 'answer' in example:
                answer = example.pop('answer')
            else:
                answer = example.pop('Answer')
            if data_source == 'aime24':
                answer = str(answer)
            question_level = example.pop('difficulty', 'unknown')
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                    "role": "system",
                    "content": system_prompt
                },
                    {
                    "role": "user",
                    "content": question_raw,
                }],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": question_raw,
                    'level': question_level
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('general-reasoner', 'train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('general-reasoner', 'test'), with_indices=True)
    gpqa_dataset = gpqa_dataset.map(function=make_map_fn('gpqa', 'test'), with_indices=True)
    aime24_dataset = aime24_dataset.map(function=make_map_fn('aime24', 'test'), with_indices=True)


    print(train_dataset)
    print(test_dataset)
    print(gpqa_dataset)
    print(aime24_dataset)
    print(aime24_dataset[0])

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    gpqa_dataset.to_parquet(os.path.join(local_dir, 'gpqa.parquet'))
    aime24_dataset.to_parquet(os.path.join(local_dir, 'aime24.parquet'))

"""
python data_preprocess/general_reasoner_science.py --dataset-name TIGER-Lab/WebInstruct-verified --local-dir data/general_reasoner_science
"""