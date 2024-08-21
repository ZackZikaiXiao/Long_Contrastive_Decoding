# --filter 参数：与之前的功能相同，用于指定过滤模型名称。
# --interactive 参数：当指定这个参数时，脚本将进入交互模式，允许你实时输入模型关键字进行过滤。
import os
import json
from prettytable import PrettyTable
import argparse
import readline

# 定义各类任务的键
tasks = {
    "单文档QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "多文档QA": ["hotpotqa", "2wikimqa", "musique"],
    "摘要": ["gov_report", "qmsum", "multi_news"],
    "Few-shot学习": ["trec", "triviaqa", "samsum"],
    "合成任务": ["passage_count", "passage_retrieval_en"],
    "代码补全": ["lcc", "repobench-p"]
}

def load_model_data(base_dir):
    all_model_data = []
    for subdir, _, files in os.walk(base_dir):
        if "result.json" in files:
            model_name = os.path.basename(subdir)
            result_path = os.path.join(subdir, "result.json")
            with open(result_path, "r") as result_file:
                results = json.load(result_file)
            stats = {}
            total_scores = []
            for task, keys in tasks.items():
                scores = [results[key] for key in keys if key in results]
                if scores:
                    average_score = sum(scores) / len(scores)
                    stats[task] = average_score
                    total_scores.append(average_score)
                else:
                    stats[task] = 0.0
            if total_scores:
                total_accuracy = sum(total_scores) / len(total_scores)
            else:
                total_accuracy = 0.0
            formatted_stats = {task: f"{stats[task]:.2f}" for task in stats}
            formatted_total_accuracy = f"{total_accuracy:.2f}"
            stat_path = os.path.join(subdir, "stat.json")
            with open(stat_path, "w") as stat_file:
                json.dump({**formatted_stats, "总准确率": formatted_total_accuracy}, stat_file, ensure_ascii=False, indent=4)
            all_model_data.append([model_name, float(formatted_total_accuracy)] + [float(formatted_stats[task]) for task in ["单文档QA", "多文档QA", "摘要", "Few-shot学习", "代码补全", "合成任务"]])
    return all_model_data

def display_table(all_model_data, filter_models):
    table = PrettyTable()
    table.field_names = ["模型", "总准确率", "单文档QA", "多文档QA", "摘要", "Few-shot学习", "代码补全", "合成任务"]
    filtered_data = [row for row in all_model_data if not filter_models or any(f in row[0] for f in filter_models)]
    filtered_data.sort(key=lambda x: x[1])
    for row in filtered_data:
        table.add_row([row[0], f"{row[1]:.2f}"] + [f"{x:.2f}" for x in row[2:]])
    print(table)

def interactive_mode(all_model_data):
    print("进入交互模式。输入模型关键字进行过滤，输入 'exit' 退出交互模式。")
    while True:
        try:
            user_input = input("请输入模型关键字：")
            if user_input.lower() == 'exit':
                break
            filter_models = user_input.split(",")
            display_table(all_model_data, filter_models)
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计模型准确率")
    parser.add_argument("--filter", type=str, help="仅统计包含指定名称的模型，多个名称用逗号分隔", default="")
    parser.add_argument("--interactive", action="store_true", help="进入交互模式")
    args = parser.parse_args()
    
    all_model_data = load_model_data("./pred")
    
    if args.interactive:
        interactive_mode(all_model_data)
    else:
        filter_models = args.filter.split(",") if args.filter else []
        display_table(all_model_data, filter_models)
