"""Defines different metrics used for evaluation of tasks."""
import functools
import numpy as np
import scipy
import math
import sklearn
import textstat
import torch
import re
from logging import getLogger
from third_party.utils import calculate_rouge, calculate_bleu, lmap
from transformers import EvalPrediction, PreTrainedTokenizer
from typing import Callable, Dict, List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from Levenshtein import ratio as lev_ratio

cola_model_name = "cointegrated/roberta-large-cola-krishna2020"

cola_tokenizer = None
cola_model = None

paraphrase_model_name = "Andrianos/paraphrase_classification_onestop_and_adversarial"

paraphrase_tokenizer = None
paraphrase_model = None

logger = getLogger(__name__)

def LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE(inputs, predictions, targets) -> dict:
    """Using the Cola already trained model, assess the percentage of linguistically accepted sentences.
    
    Note: Takes 3 (instead of 1) inputs for compatibility reasons.
    """
    global cola_model, cola_tokenizer
    if cola_model is None or cola_tokenizer is None:
        print('downloading Cola Model')
        cola_tokenizer = AutoTokenizer.from_pretrained(cola_model_name)
        cola_model = AutoModelForSequenceClassification.from_pretrained(cola_model_name)
        cola_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Tokenize all texts.
    model_inputs = cola_tokenizer(predictions, return_tensors="pt", truncation=True, padding=True)
    model_inputs = {name: tensor.to(cola_model.device) for name, tensor in model_inputs.items()}

    with torch.no_grad():
        logits = cola_model(**model_inputs)[0]

    predicted_classes = torch.argmax(logits, dim=-1)
    # Count the number of instances where the predicted class is 0 (grammatically correct)
    count_class_0 = (predicted_classes == 0).sum().item()
    # Calculate the percentage of predicted class being 1
    percentage_class_0 = (count_class_0 / len(predicted_classes)) * 100

    return {"LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE": percentage_class_0}


def CORRECT_PARAPHRASE_PERCENTAGE(inputs, predictions, targets) -> dict:
    """Using the model trained in another part of this project, access the percentage of correct paraphrases.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons
    """
    global paraphrase_model, paraphrase_tokenizer
    if paraphrase_model is None or paraphrase_tokenizer is None:
        print('downloading Paraphrase Model')
        paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)
        paraphrase_model = AutoModelForSequenceClassification.from_pretrained(paraphrase_model_name)
        paraphrase_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Tokenize the sentences in this batch and convert them to tensors 
    model_inputs = paraphrase_tokenizer(inputs, predictions, padding=True, truncation=True, max_length=512, return_tensors="pt")
    model_inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in model_inputs.items()}

    # Get the model's output for this batch
    with torch.no_grad():
        logits = paraphrase_model(**model_inputs)[0]

    predicted_classes = torch.argmax(logits, dim=-1)
    # Count the number of instances where the predicted class is 1 (grammatically correct)
    count_class_1 = (predicted_classes == 1).sum().item()
    # Calculate the percentage of predicted class being 1
    percentage_class_1 = (count_class_1 / len(predicted_classes)) * 100

    return {"CORRECT_PARAPHRASE_PERCENTAGE": percentage_class_1}

def AVG_GFI_SRC_PRED_DIFF(inputs, predictions, targets) -> dict:
    """Pairwise (SRC-PRED) difference Gunning Fog Index (GFI) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons
    """
    input_scores = np.array([textstat.gunning_fog(input) for input in inputs])
    pred_scores = np.array([textstat.gunning_fog(pred) for pred in predictions])
    return {"AVG_GFI_SRC_PRED_DIFF": np.mean(input_scores - pred_scores)}

def AVG_FRE_SRC_PRED_DIFF(inputs, predictions, targets) -> dict:
    """Pairwise (SRC-PRED) difference Flesch Reading Ease (FRE) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    input_scores = np.array([textstat.flesch_reading_ease(input) for input in inputs])
    pred_scores = np.array([textstat.flesch_reading_ease(pred) for pred in predictions])
    return {"AVG_FRE_SRC_PRED_DIFF": np.mean(input_scores - pred_scores)}

def AVG_FKGL_SRC_PRED_DIFF(inputs, predictions, targets) -> dict:
    """Pairwise (SRC-PRED) difference Flesch-Kincaid Grade Level (FKGL) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    input_scores = np.array([textstat.flesch_kincaid_grade(input) for input in inputs])
    pred_scores = np.array([textstat.flesch_kincaid_grade(pred) for pred in predictions])
    return {"AVG_FKGL_SRC_PRED_DIFF": np.mean(input_scores - pred_scores)}

def AVG_ARI_SRC_PRED_DIFF(inputs, predictions, targets) -> dict:
    """Pairwise (SRC-PRED) difference Automated Readability Index (ARI) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    input_scores = np.array([textstat.automated_readability_index(input) for input in inputs])
    pred_scores = np.array([textstat.automated_readability_index(pred) for pred in predictions])
    return {"AVG_ARI_SRC_PRED_DIFF": np.mean(input_scores - pred_scores)}

def AVG_DCRF_SRC_PRED_DIFF(inputs, predictions, targets) -> dict:
    """Pairwise (SRC-PRED) difference Dale-Chall Readability Formula (DCRF) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    input_scores = np.array([textstat.dale_chall_readability_score(input) for input in inputs])
    pred_scores = np.array([textstat.dale_chall_readability_score(pred) for pred in predictions])
    return {"AVG_DCRF_SRC_PRED_DIFF": np.mean(input_scores - pred_scores)}

def AVG_SMOG_SRC_PRED_DIFF(inputs, predictions, targets) -> dict:
    """Pairwise (SRC-PRED) difference Simple Measure of Gobbledygook (SMOG) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    input_scores = np.array([textstat.smog_index(input) for input in inputs])
    pred_scores = np.array([textstat.smog_index(pred) for pred in predictions])
    return {"AVG_SMOG_SRC_PRED_DIFF": np.mean(input_scores - pred_scores)}

def AVG_ASL_SRC_PRED_DIFF(inputs, predictions, targets) -> dict:
    """Pairwise (SRC-PRED) difference Average Sentence Length (ASL) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    input_scores = np.array([textstat.avg_sentence_length(input) for input in inputs])
    pred_scores = np.array([textstat.avg_sentence_length(pred) for pred in predictions])
    return {"AVG_ASL_SRC_PRED_DIFF": np.mean(input_scores - pred_scores)}

def AVG_GFI_PRED_TGT_ABSDIFF(inputs, predictions, targets) -> dict:
    """Pairwise (PRED-TGT) absolute difference Gunning Fog Index (GFI) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons
    """
    pred_scores = np.array([textstat.gunning_fog(pred) for pred in predictions])
    target_scores = np.array([textstat.gunning_fog(target) for target in targets])
    return {"AVG_GFI_PRED_TGT_ABSDIFF": np.mean(np.abs(pred_scores - target_scores))}

def AVG_FRE_PRED_TGT_ABSDIFF(inputs, predictions, targets) -> dict:
    """Pairwise (PRED-TGT) absolute difference Flesch Reading Ease (FRE) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    pred_scores = np.array([textstat.flesch_reading_ease(pred) for pred in predictions])
    target_scores = np.array([textstat.flesch_reading_ease(target) for target in targets])
    return {"AVG_FRE_PRED_TGT_ABSDIFF": np.mean(np.abs(pred_scores - target_scores))}

def AVG_FKGL_PRED_TGT_ABSDIFF(inputs, predictions, targets) -> dict:
    """Pairwise (PRED-TGT) absolute difference Flesch-Kincaid Grade Level (FKGL) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    pred_scores = np.array([textstat.flesch_kincaid_grade(pred) for pred in predictions])
    target_scores = np.array([textstat.flesch_kincaid_grade(target) for target in targets])
    return {"AVG_FKGL_PRED_TGT_ABSDIFF": np.mean(np.abs(pred_scores - target_scores))}

def AVG_ARI_PRED_TGT_ABSDIFF(inputs, predictions, targets) -> dict:
    """Pairwise (PRED-TGT) absolute difference Automated Readability Index (ARI) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    pred_scores = np.array([textstat.automated_readability_index(pred) for pred in predictions])
    target_scores = np.array([textstat.automated_readability_index(target) for target in targets])
    return {"AVG_ARI_PRED_TGT_ABSDIFF": np.mean(np.abs(pred_scores - target_scores))}

def AVG_DCRF_PRED_TGT_ABSDIFF(inputs, predictions, targets) -> dict:
    """Pairwise (PRED-TGT) absolute difference Dale-Chall Readability Formula (DCRF) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    pred_scores = np.array([textstat.dale_chall_readability_score(pred) for pred in predictions])
    target_scores = np.array([textstat.dale_chall_readability_score(target) for target in targets])
    return {"AVG_DCRF_PRED_TGT_ABSDIFF": np.mean(np.abs(pred_scores - target_scores))}

def AVG_SMOG_PRED_TGT_ABSDIFF(inputs, predictions, targets) -> dict:
    """Pairwise (PRED-TGT) absolute difference Simple Measure of Gobbledygook (SMOG) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    pred_scores = np.array([textstat.smog_index(pred) for pred in predictions])
    target_scores = np.array([textstat.smog_index(target) for target in targets])
    return {"AVG_SMOG_PRED_TGT_ABSDIFF": np.mean(np.abs(pred_scores - target_scores))}

def AVG_ASL_PRED_TGT_ABSDIFF(inputs, predictions, targets) -> dict:
    """Pairwise (PRED-TGT) absolute difference Average Sentence Length (ASL) score.
    
    Note: Takes 3 (instead of 2) inputs for compatibility reasons.
    """
    pred_scores = np.array([textstat.avg_sentence_length(pred) for pred in predictions])
    target_scores = np.array([textstat.avg_sentence_length(target) for target in targets])
    return {"AVG_ASL_PRED_TGT_ABSDIFF": np.mean(np.abs(pred_scores - target_scores))}

def PRED_ONLY_GFI(inputs, predictions, targets) -> dict:
    """Target only Average Gunning Average Fog Index (GFI) score.
    
    Note: Takes 3 inputs for compatibility reasons.
    """
    scores = [textstat.gunning_fog(pred) for pred in predictions]
    return {"PRED_ONLY_GFI": np.mean(scores)}

def PRED_ONLY_FRE(inputs, predictions, targets) -> dict:
    """Target only Average Flesch Reading Ease (FRE) score.
    
    Note: Takes 3 inputs for compatibility reasons.
    """
    scores = [textstat.flesch_reading_ease(pred) for pred in predictions]
    return {"PRED_ONLY_FRE": np.mean(scores)}

def PRED_ONLY_FKGL(inputs, predictions, targets) -> dict:
    """Target only Average Flesch-Kincaid Grade Level (FKGL) score.
    
    Note: Takes 3 inputs for compatibility reasons.
    """
    scores = [textstat.flesch_kincaid_grade(pred) for pred in predictions]
    return {"PRED_ONLY_FKGL": np.mean(scores)}

def PRED_ONLY_ARI(inputs, predictions, targets) -> dict:
    """Target only Average Automated Readability Index (ARI) score.
    
    Note: Takes 3 inputs for compatibility reasons.
    """
    scores = [textstat.automated_readability_index(pred) for pred in predictions]
    return {"PRED_ONLY_ARI": np.mean(scores)}

def PRED_ONLY_DCRF(inputs, predictions, targets) -> dict:
    """Target only Average Dale-Chall Readability Formula (DCRF) score.
    
    Note: Takes 3 inputs for compatibility reasons.
    """
    scores = [textstat.dale_chall_readability_score(pred) for pred in predictions]
    return {"PRED_ONLY_DCRF": np.mean(scores)}

def PRED_ONLY_SMOG(inputs, predictions, targets) -> dict:
    """Target only Average Simple Measure of Gobbledygook (SMOG) score.
    
    Note: Takes 3 inputs for compatibility reasons.
    """
    scores = [textstat.smog_index(pred) for pred in predictions]
    return {"PRED_ONLY_SMOG": np.mean(scores)}

def PRED_ONLY_ASL(inputs, predictions, targets) -> dict:
    """Target only (Averaged) Average Sentence Length (ASL) score.
    
    Note: Takes 3 inputs for compatibility reasons.
    """
    scores = [textstat.avg_sentence_length(pred) for pred in predictions]
    return {"PRED_ONLY_ASL": np.mean(scores)}

def SRC_PRED_EXACT_COPY_PERCENTAGE(inputs, predictions, targets) -> dict:
    """Calculate the percentage of exact copies in predictions with respect to source inputs.

    Note: Takes 3 inputs for compatibility reasons.
    """
    exact_copies = sum(pred == input for pred, input in zip(predictions, inputs))
    return {"SRC_PRED_EXACT_COPY_PERCENTAGE": 100 * exact_copies / len(predictions) if predictions else 0}

def AVG_PRED_TGT_LEV_DIST(inputs, predictions, targets) -> dict:
    """Calculate the average normalised Levenshtein distance between predictions and targets.

    Note: Takes 3 inputs for compatibility reasons.
    """
    ratios = []
    for pred, tgt in zip(predictions, targets):
        ratios.append(lev_ratio(pred,tgt))
    return {"AVG_PRED_TGT_LEV_DIST": (sum(ratios) / len(ratios)) if predictions else 0}


def rouge(inputs, predictions, targets) -> dict:
    """Computes rouge score."""
    return calculate_rouge(predictions, targets)


def bleu(predictions, targets) -> dict:
    """Computes bleu score."""
    return calculate_bleu(predictions, targets)


def accuracy(predictions, targets) -> dict:
    """Computes the average accuracy."""
    return {"acc": 100 * ((np.array(predictions) == np.array(targets)).mean())}


def pearson_corrcoef(predictions, targets) -> dict:
    """Computes Pearson correlation coefficient."""
    pearson_corrcoef = 100 * scipy.stats.pearsonr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    return {"pearson_corrcoef": pearson_corrcoef}


def spearman_corrcoef(predictions, targets) -> dict:
    """Computes Spearman correlation coefficient."""
    spearman_corrcoef = 100 * scipy.stats.spearmanr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    return {"spearman_corrcoef": spearman_corrcoef}


def f1_score_with_invalid(predictions, targets) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      targets: list of targets, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions.
    invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
    # For any prediction != 0 or 1, we set the prediction to the opposite of its corresponding target.
    predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}


# TODO: maybe gaurd against invalid values https://stackoverflow.com/questions/56865344/how-do-i-calculate-the-matthews-correlation-coefficient-in-tensorflow
def matthews_corrcoef(predictions, targets) -> dict:
    """Computes the Matthews correlation coefficient."""
    return {"mcc": 100 * sklearn.metrics.matthews_corrcoef(targets, predictions)}


def build_compute_metrics_fn(task_names: List[str],
                             tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    """Builds a dictionary from each task to the task metric."""

    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        input_array = np.where(pred.inputs == -100, 0, pred.inputs) # 
        if pred.inputs is not None:
            input_str = tokenizer.batch_decode(input_array, skip_special_tokens=True)
            input_str = lmap(str.strip, input_str)
        else:
            input_str = None
        return input_str, pred_str, label_str

    def compute_metrics(pred: EvalPrediction, metrics, post_processor=None) -> Dict:
        input_str, pred_str, label_str = decode_pred(pred) # checked for validity once already.
        # Applies task post-processor.
        if post_processor is not None:
            input_str = [post_processor(pred) for pred in pred_str] # not sure if necessary, irrelevant anyway.
            pred_str = [post_processor(pred) for pred in pred_str]
            label_str = [post_processor(label) for label in label_str]
        if input_str is not None:
                prefixes = [
                'Simplify from advanced to intermediate: ',
                'Simplify from advanced to elementary: ',
                'Simplify from intermediate to elementary: ',
                'Rewrite from elementary to intermediate: ',
                'Rewrite from elementary to advanced: ',
                'Rewrite from intermediate to advanced: ',
                ]
                # Boolean variable to track if a prefix was found in the current string.
                prefix_found = True

                # Loop over each string in the input list.
                for i in range(len(input_str)):
                    # Initially set prefix_found to False for each new string.
                    prefix_found = False
                    # Loop over each prefix.
                    for prefix in prefixes:
                        # Check if the current string starts with the current prefix.
                        if input_str[i].startswith(prefix):
                            # If it does, remove the prefix and set prefix_found to True.
                            input_str[i] = re.sub('^' + prefix, '', input_str[i])
                            prefix_found = True
                            # Break the inner loop as we have found a prefix in the current string.
                            break
                    if not prefix_found: # If no prefix was found after checking all prefixes, no need to check all inputs.
                        break
                if prefix_found:
                    print('Removed all prefixes from strings') # else nothing prints.
        eval_results = {}
        for metric in metrics:
            if input_str is not None:
                eval_results.update(metric(input_str, pred_str, label_str))
            else:
                eval_results.update(metric(pred_str, label_str))
            if metric.__name__ in ['bleu', 'rouge']:
                gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
                eval_results.update({"gen_len": gen_len})
        return eval_results

    def tasks_metrics(task) -> Dict:
        from data.tasks import TASK_MAPPING
        from data.postprocessors import get_post_processor
        if task.startswith('onestop_parallel_sentence'):
            task = 'onestop_parallel_sentence_'
        elif task.startswith('onestop_parallel_text'):
            task = 'onestop_parallel_text_'
        else:
            task = task
        return functools.partial(compute_metrics, metrics=TASK_MAPPING[task].metrics,
                                 post_processor=get_post_processor(task))

    return {task: tasks_metrics(task) for task in task_names}
