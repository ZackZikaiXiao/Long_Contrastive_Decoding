import json
import re
from tqdm import tqdm

def string_match_all(preds, refs, verbose=0):
    total_precision = 0.0
    total_recall = 0.0
    total_f1_score = 0.0
    num_preds = len(preds)

    for pred_list, ref_list in zip(preds, refs):
        matched_ref_count = 0
        for ref_item in ref_list:
            for pred_item in pred_list:
                if ref_item.lower() == pred_item.lower():
                    matched_ref_count += 1.0
                    break
        
        recall = matched_ref_count / len(ref_list)
        precision = matched_ref_count / len(pred_list)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score

        if verbose:
            # Debug print statements
            print(f'Pred List: {pred_list}')
            print(f'Ref List: {ref_list}')
            print(f'Matched Ref Count: {matched_ref_count}')
            print(f'Recall: {recall}')
            print(f'Precision: {precision}')
            print(f'F1 Score: {f1_score}')
    
    avg_precision = total_precision / num_preds
    avg_recall = total_recall / num_preds
    avg_f1_score = total_f1_score / num_preds

    if verbose:
        print(f'Average Precision: {avg_precision}')
        print(f'Average Recall: {avg_recall}')
        print(f'Average F1 Score: {avg_f1_score}')
    
    result =  {
        'precision': round(avg_precision * 100, 2),
        'recall': round(avg_recall * 100, 2),
        'f1_score': round(avg_f1_score * 100, 2)
    }
    return result['f1_score']


def read_manifest(predictions_file):
    with open(predictions_file, 'r') as f:
        return [json.loads(line) for line in f]

def get_pred_and_ref(predictions_file: str, task_config: dict, prediction_field: str = 'prediction', references_field: str = 'ground_truth'):
    lines = read_manifest(predictions_file)

    predicts = []
    references = []

    for line in tqdm(lines):
        predict = postprocess_pred(line[prediction_field], task_config)
        reference = line.get(references_field, [line.get('ground_truth', '')])
        
        predicts.append(predict)
        references.append(reference)
        
    return predicts, references

def postprocess_pred(predict_str: str, task_config: dict):
    predict_str = re.sub(r'[\x00-\x1f]', '\n', predict_str.strip())
    return [p.strip() for p in predict_str.split(',')]

def run_evaluation_per_task(task_config: dict, predictions_file: str, verbose: int = 0):
    predicts, references = get_pred_and_ref(predictions_file, task_config)

    task_nulls = f'{sum(len(x) == 0 for x in predicts)}/{len(predicts)}'
    task_score = task_config['metric_fn'](predicts, references) if references and references[0][0] is not None else 0.0

    if verbose:
        print('=' * 40)
        for i, (reference, predict) in enumerate(zip(references, predicts)):
            print(f'Reference : {reference}')
            print(f'Prediction: {predict}')
            print('=' * 40)
            if i >= verbose:
                break

    return task_score, task_nulls, predicts

def evaluate_predictions(predictions_file: str, verbose: int = 0):
    task_config = {'metric_fn': string_match_all}
    task_score, task_nulls, predicts = run_evaluation_per_task(task_config, predictions_file, verbose=verbose)
    return task_score, task_nulls

if __name__ == "__main__":
    predictions_file = '/home/zikaixiao/zikaixiao/LongLoRA-main/benchmark/super_retrieval/results/llama-3-8B-8k/preds_variable_tracking_4k.jsonl'
    task_score, task_nulls = evaluate_predictions(predictions_file, verbose=0)
    print(f'Task Score: {task_score}')
    print(f'Task Nulls: {task_nulls}')