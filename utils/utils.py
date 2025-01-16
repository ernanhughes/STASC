import sys

sys.path.append('../')
sys.path.append('./')

from utils.eval_utils import has_answer


def KM(ds, target_col, gt_col):
    total_score = 0
    total_count = 0
    for sample in ds:
        corr_ans = sample[gt_col]
        model_ans = sample[target_col]

        if not isinstance(model_ans, list):
            # Single prediction case
            scores = [has_answer(corr_ans, model_ans)]
        elif isinstance(model_ans[0], list):
            # model_ans is a list of lists
            scores = [has_answer(corr_ans, sub_ans) for sub_list in model_ans for sub_ans in sub_list]
        else:
            # model_ans is a list of predictions
            scores = [has_answer(corr_ans, ans) for ans in model_ans]

        # Accumulate scores and counts
        total_score += sum(scores)
        total_count += len(scores)
    
    return total_score / total_count if total_count > 0 else 0.0


def add_metric(sample, target_col, gt_col, metric, metric_name):
    corr_ans = sample[gt_col]
    model_ans = sample[target_col]
    is_corr = metric(corr_ans, model_ans)
    sample[f"{metric_name}_{target_col}"] = is_corr
    return sample




def flatten_predictions(predictions):
    if not isinstance(predictions, list):
        return [predictions]
    # Recursively flatten the list
    flattened = []
    for pred in predictions:
        if isinstance(pred, list):
            flattened.extend(flatten_predictions(pred))
        else:
            flattened.append(pred)
    return flattened