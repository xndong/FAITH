import logging
import re
import os
import string
import argparse

from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default="llama3", help="Model name.", choices=model_path_dict.keys())
parser.add_argument('--model_suffix', type=str, default="train_sft_base_bnb_icl", help="Model suffix.")
parser.add_argument('--dataset', type=str, default="sciq", help="Dataset name.", choices=dataset_list)
parser.add_argument('--data_file', type=str, default="validation", help="data split name.", choices=["validation", "train"])
parser.add_argument('--train_type', type=str) # , choices=["ppo", "sft"], required=True,
parser.add_argument('--steps', type=str, help="File suffix to save the eval log.")
parser.add_argument('--score_dir', type=str, default="./data/{}/prep/") # data_bake_k6
parser.add_argument('--input_dir', type=str, default="/gfshome/UAlign/data_xn-validation-results")
parser.add_argument('--score_use', action="store_true", help="Whether to use the score file.")

args = parser.parse_args()


"""
Evaluate QA outputs: TriviaQA, WebQA
"""
# Normalize the answer.
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    # return white_space_fix(remove_punc(lower(s)))
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Compute the exact match score in different ways.
def compute_exact(a_gold, a_pred):
    eval_type = "EM_RP"

    if eval_type == "EM":
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))
    elif eval_type == "EM_R":
        return int(normalize_answer(a_gold) in normalize_answer(a_pred))
    elif eval_type == "EM_P":
        return int(normalize_answer(a_pred) in normalize_answer(a_gold))
    elif eval_type == "EM_RP":
        return int(normalize_answer(a_gold) in normalize_answer(a_pred)) or int(normalize_answer(a_pred) in normalize_answer(a_gold))


"""
Evaluate part
"""
def compute_sample_score(label, output, dataset):
    # import pdb; pdb.set_trace()
    score = compute_exact(label, output)

    return score


def compute_scores(outputs, decoding="greedy", gold_answer=None, dataset=None):
    # For greedy decoding answers
    scores = []
    for output in outputs:
        if decoding == "greedy":
            greedy_answer = output["greedy_decoding"]
            greedy_score = compute_sample_score(gold_answer, greedy_answer, dataset)
            scores.append(greedy_score)
            average_score = sum(scores) / len(scores)
        # For sampling answers
        else:
            sample_scores = []
            # import pdb; pdb.set_trace()
            for sample_output in output["temperature_sampling"]:
                sample_score = compute_sample_score(gold_answer, sample_output, dataset)
                sample_scores.append(sample_score)

            scores.append(sample_scores)
            average_score = sum(map(sum, scores)) / (len(scores) * len(scores[0]))

    return scores, average_score


def evaluate():
    # Format output file.
    args.score_dir = os.path.join(args.score_dir.format(args.dataset), "{}_{}_no_lora_icl". \
                                   format(args.model_name, args.dataset), "{}_{}_{}". \
                                    format(args.model_name, args.dataset, args.data_file))
    log_path = os.path.join(args.input_dir, f"eval-{args.train_type}_{args.steps}-{args.model_name}-{args.dataset}.log")
    if not os.path.exists(args.input_dir):
        os.mkdir(args.input_dir)
    
    # print(log_path)
    # Set logging.
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    # Loading dataset.
    data_path = os.path.join(args.input_dir, f"{args.model_name}_{args.dataset}_{args.data_file}_results.json")
    data_pool = read_json(data_path)
    
    logging.info(f"Load data from {data_path} ...")

    if args.score_use:
        score_path = os.path.join(f"{args.score_dir}.json")
        score_list = read_json(score_path)

        true, knowns, unknowns, known_correct, unknown_correct = 0, 0, 0, 0, 0
            
        for idx, data in enumerate(data_pool):
            assert data["question_id"] == score_list[idx]["question_id"]
            score = compute_sample_score(label=data["answer"], output=data["output"], dataset=args.dataset)
            true += score

            if score:
                if score_list[idx]["scores"]["greedy_scores_avg"] > 0:
                    known_correct += 1
            else:
                if score_list[idx]["scores"]["greedy_scores_avg"] == 0 and "sorry" in data["output"].lower():
                    unknown_correct += 1

            if score_list[idx]["scores"]["greedy_scores_avg"] > 0:
                knowns += 1
            else:
                unknowns += 1

            data["known"] = score_list[idx]["scores"]["greedy_scores_avg"]


        logging.info(f"Results:\n \
            Total: {knowns + unknowns},\n \
            Known: {knowns},\n \
            Known Correct: {known_correct},\n \
            Unknown: {unknowns},\n \
            Unknown Correct: {unknown_correct},\n \
            True Accuracy: {round(known_correct/(knowns+unknowns), 4)},\n \
            Precision: {round(known_correct/knowns, 4)},\n \
            Truth: {round((known_correct+unknown_correct)/(knowns+unknowns), 4)},\n \
            Accuracy: {round(true/len(data_pool), 4)}"
        )
        
        write_json(data_path, data_pool)
    else:
        score = 0
        for data in data_pool:
            score += compute_sample_score(label=data["answer"], output=data["output"], dataset=args.dataset)

        logging.info(f"Total Accuracy: {score/len(data_pool)}")


if __name__=="__main__":
    evaluate()


# python eval.py --model_name mistral --dataset sciq --data_file validation --score_use
# python eval.py --model_name mistral --dataset triviaqa --data_file validation --score_use
# python eval.py --model_name mistral --dataset nqopen --data_file validation --score_use