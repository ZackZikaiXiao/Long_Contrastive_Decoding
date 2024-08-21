import os
import json
import argparse
import numpy as np
from stat_e import range_avg

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama-3-8B-262k_rope_permu")
    parser.add_argument('--longbench_dir', type=str, default="/home/zikaixiao/zikaixiao/LongLoRA-main/benchmark/LongBench", help="Directory to save the evaluation results")
    parser.add_argument('--e', action='store_true', default=False, help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# def scorer_e(dataset, predictions, answers, lengths, all_classes):
#     # scores = {"0-4k": [], "4-8k": [], "8k+": []}
#     scores = {"0-16k":[], "16k+": []}
#     for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
#         score = 0.
#         length *= 5.5 # 统计来看，去空格的Characters是Words的5.5倍
#         if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
#             prediction = prediction.lstrip('\n').split('\n')[0]
#         for ground_truth in ground_truths:
#             score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
#         if length < 16000:
#             scores["0-16k"].append(score)
#         else:
#             scores["16k+"].append(score)

#     for key in scores.keys():
#         scores[key] = round(100 * np.mean(scores[key]), 2)
#     return scores

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-2k": [], "2k-8k": [], "8k-16k": [], "16k-40k": [], "40k-72k": [], "72k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        length *= 5.5  # 统计来看，去空格的Characters是Words的5.5倍
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 2000:
            scores["0-2k"].append(score)
        elif length < 8000:
            scores["2k-8k"].append(score)
        elif length < 16000:
            scores["8k-16k"].append(score)
        elif length < 40000:
            scores["16k-40k"].append(score)
        elif length < 72000:
            scores["40k-72k"].append(score)
        else:
            scores["72k+"].append(score)

    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    longbench_dir = args.longbench_dir
    scores = dict()
    if args.e:
        path = f"{longbench_dir}/pred_e/{args.model_name}/"
    else:
        path = f"{longbench_dir}/pred/{args.model_name}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    if args.e:
        out_path = f"{longbench_dir}/pred_e/{args.model_name}/result.json"
    else:
        out_path = f"{longbench_dir}/pred/{args.model_name}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    if args.e:
        range_avg(out_path)
