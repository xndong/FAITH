import logging
import re
import os
import string
import argparse
from collections import Counter

from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default="llama3", help="Model name.", choices=model_path_dict.keys())
parser.add_argument('--model_suffix', type=str, default="train_sft_base_bnb_icl", help="Model suffix.")
parser.add_argument('--dataset', type=str, default="triviaqa", help="Dataset name.", choices=dataset_list)
parser.add_argument('--data_file', type=str, default="validation", help="Data file name.", choices=["validation", "train"])
parser.add_argument('--data_suffix', type=str, default="1k_8s", help="File name to save the results.")
parser.add_argument('--input_dir', type=str, default="./data/{}/prep/")
parser.add_argument('--output_dir', type=str, default="./data/{}/prep/")
parser.add_argument('--input_suffix', type=str, default="no_lora", help="对应sample.py中的model_suffix")

args = parser.parse_args()


def calculate_probabilities(incorrect_ans_list):
    if incorrect_ans_list == []:
        return []
    # Count the number of occurrences of each answer
    noun_counts = Counter(incorrect_ans_list)                   #REVIEW: 对错误的回答做Counter统计
    total_nouns = sum(noun_counts.values())
    
    # Calculate the probability of each answer
    probabilities = {noun: count / total_nouns for noun, count in noun_counts.items()}
    
    return probabilities

def calculate_entropy(answer_list, greedy_scores):
    """Calculate the entropy of a probability distribution."""
    conf_score = calculate_confidence(greedy_scores)
    incorrect_ans_list = []
    for idx, ans in enumerate(answer_list):
        if greedy_scores[idx] == 0:
            incorrect_ans_list.append(ans["greedy_decoding"])
    # print(incorrect_ans_list)

    # import pdb; pdb.set_trace()
    if incorrect_ans_list == []:
        probabilities = [conf_score]
    else:
        probabilities = list(calculate_probabilities(incorrect_ans_list).values())*(1-conf_score) #REVIEW: 对所有数值乘以(1-conf_score)——因为后面要加入一个[conf_score]——保证sum起来是1.0
        probabilities = probabilities + [conf_score]
    # print(probabilities)
    # Ensure probabilities sum to 1
    if not math.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    
    # Calculate the entropy
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    return entropy


def calculate_confidence(probabilities):
    """Calculate the confidence of a probability distribution."""
    return max(probabilities)


def uncertainty_estimations():
    """Uncertainty estimations on QA outputs."""
    # Format input and output files.
    args.input_file = os.path.join(args.input_dir.format(args.dataset), 
                                   "{}_{}_{}_icl".format(args.model_name, args.dataset, args.input_suffix), 
                                   "{}_{}_{}.json".format(args.model_name, args.dataset, args.data_file)
                                   )
    args.conf_dir = os.path.join(args.input_dir.format(args.dataset), 
                                 "{}_{}_confidence".format(args.model_name, args.dataset)
                                 )
    args.entro_dir = os.path.join(args.input_dir.format(args.dataset), 
                                  "{}_{}_entropy".format(args.model_name, args.dataset)
                                  )

    if not os.path.exists(args.conf_dir):
        os.makedirs(args.conf_dir)
    if not os.path.exists(args.entro_dir):
        os.makedirs(args.entro_dir)

    conf_file = args.conf_dir + "/{}_{}_{}.json".format(args.model_name, args.dataset, args.data_file)  
    entro_file = args.entro_dir + "/{}_{}_{}.json".format(args.model_name, args.dataset, args.data_file)

    # Load the QA dataset.
    dataset = read_json(args.input_file)

    conf_list, entro_list = [], []

    for data in dataset: #NOTE: sciq len 11679
        # import pdb; pdb.set_trace()
        # print(data["question_id"])
        # confidence = calculate_confidence(data["scores"]["greedy_scores"])
        confidence = data["scores"]["greedy_scores_avg"]    # REVIEW: average scores, confidence用average, entropy仍然用的最大值
        conf_data = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": confidence
        }                                       # entry
        conf_list.append(conf_data)             # entry list
        data["confidence"] = confidence # 更新dataset的entry,新增一个confidence项(e.g. data/sciq/prep/llama3_sciq_no_lora_icl/llama3_sciq_train.json)

        # import pdb; pdb.set_trace()
        entropy = calculate_entropy(data["outputs"], data["scores"]["greedy_scores"])
        entro_data = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": entropy
        }                                        # entry     
        entro_list.append(entro_data)            # entry list
        data["entropy"] = entropy       # 更新dataset的entry,新增一个entropy项

    # Save the confidence and entropy estimations.
    write_json(conf_file, conf_list)            # dump entry list to conf_file
    write_json(entro_file, entro_list)          # dump entry list to entro_file
    write_json(args.input_file, dataset)        # 将更新过的数据集写回


def reward_dataset():
    """Reward dataset on QA outputs."""
    # Format input and output files.
    args.input_file = os.path.join(args.input_dir.format(args.dataset), 
                                   "{}_{}_{}_icl".format(args.model_name, args.dataset, args.input_suffix), 
                                   "{}_{}_{}.json".format(args.model_name, args.dataset, args.data_file)
                                   )
    args.reward_dir = os.path.join(args.input_dir.format(args.dataset), 
                                   "{}_{}_reward".format(args.model_name, args.dataset)
                                   )

    if not os.path.exists(args.reward_dir):
        os.makedirs(args.reward_dir)

    reward_file = args.reward_dir + "/{}_{}_{}.json".format(args.model_name, args.dataset, args.data_file)  

    # Load the QA dataset.
    dataset = read_json(args.input_file)
    reward_list = []

    for data in dataset: #NOTE: sciq len 11679
        question = data["question"]
        confidence = data["confidence"]
        entropy = data["entropy"]
        labels = data["scores"]["greedy_scores"]

        # import pdb; pdb.set_trace()
        for label, output in zip(labels, data["outputs"]):
            context = question + f"\n### Conf ###: {confidence}" + \
            f"\n### Entro ###: {entropy}\n" + output["greedy_decoding"]

            reward_data = {
                "question": context,
                "answer": label
            }
            reward_list.append(reward_data)

    # Save the reward dataset.
    write_json(reward_file, reward_list)


def align_dataset():
    """Alignment dataset on QA outputs."""
    # Format input and output files.
    args.input_file = os.path.join(args.input_dir.format(args.dataset), 
                                   "{}_{}_{}_icl".format(args.model_name, args.dataset, args.input_suffix), 
                                   "{}_{}_{}.json".format(args.model_name, args.dataset, args.data_file)
                                   )
    args.align_dir = os.path.join(args.input_dir.format(args.dataset), 
                                  "{}_{}_align".format(args.model_name, args.dataset)
                                  )

    if not os.path.exists(args.align_dir):
        os.makedirs(args.align_dir)

    align_file = args.align_dir + "/{}_{}_{}.json".format(args.model_name, args.dataset, args.data_file)  

    # Load the QA dataset.
    dataset = read_json(args.input_file)
    align_list = []

    for data in dataset: #NOTE: sciq len 11679
        question_id = data["question_id"]
        question = data["question"]
        answer = data["answer"]
        confidence = data["confidence"]
        entropy = data["entropy"]

        context = question + f"\n### Conf ###: {confidence}" + \
            f"\n### Entro ###: {entropy}"

        # import pdb; pdb.set_trace()
        align_data = {
            "question_id": question_id,
            "question": context,
            "answer": answer
        }
        align_list.append(align_data)
        
    # Save the align dataset.
    write_json(align_file, align_list)


if __name__ == "__main__":
    uncertainty_estimations()
    reward_dataset()
    align_dataset()
