import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Read the content of the log files

file_names = [
    'small_separate_readability_init_Lhypernet.log',
    'small_original_hyperformer++.log',
]

task_prefix = 'onestop_parallel_sentence_'
log_contents = []
for file_name in file_names:
    with open(file_name, 'r') as file:
        log_contents.append(file.read())


new_metric_suffixes = [
    "SARI",
    "LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE",
    "CORRECT_PARAPHRASE_PERCENTAGE",
    "AVG_PRED_TGT_LEV_DIST",
    "AVG_GFI_SRC_PRED_DIFF",
    "AVG_FRE_SRC_PRED_DIFF",
    "AVG_FKGL_SRC_PRED_DIFF",
    "AVG_ARI_SRC_PRED_DIFF",
    "AVG_DCRF_SRC_PRED_DIFF",
    "AVG_SMOG_SRC_PRED_DIFF",
    "AVG_ASL_SRC_PRED_DIFF",
    "AVG_GFI_PRED_TGT_ABSDIFF",
    "AVG_FRE_PRED_TGT_ABSDIFF",
    "AVG_FKGL_PRED_TGT_ABSDIFF",
    "AVG_ARI_PRED_TGT_ABSDIFF",
    "AVG_DCRF_PRED_TGT_ABSDIFF",
    "AVG_SMOG_PRED_TGT_ABSDIFF",
    "AVG_ASL_PRED_TGT_ABSDIFF",
    "PRED_ONLY_GFI",
    "PRED_ONLY_FRE",
    "PRED_ONLY_FKGL",
    "PRED_ONLY_ARI",
    "PRED_ONLY_DCRF",
    "PRED_ONLY_SMOG",
    "PRED_ONLY_ASL",
    "SRC_PRED_EXACT_COPY_PERCENTAGE",
    "loss", 
    "rouge1", 
    "rouge2", 
    "rougeL",
]


primary_metrics = [
    "SARI",
    "LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE",
    "CORRECT_PARAPHRASE_PERCENTAGE",
    "PRED_ONLY_GFI",
    "PRED_ONLY_FRE",
    "PRED_ONLY_FKGL",
    "PRED_ONLY_ARI",
    "SRC_PRED_EXACT_COPY_PERCENTAGE",
    "loss", 
    "rougeL",
]

secondary_metrics = [metric for metric in new_metric_suffixes if metric not in primary_metrics]

metric_sets = {
    'primary': primary_metrics,
    'secondary': secondary_metrics
}

metric_suffixes = new_metric_suffixes

if task_prefix.startswith('onestop'):
    combinations = [
    (first, second)
    for first in ['adv', 'int', 'ele']
    for second in ['adv', 'int', 'ele']
    if first != second
    ]
else:
    combinations = [
    (first, second)
    for first in ['Level0', 'Level1', 'Level3']
    for second in ['Level0', 'Level1', 'Level3']
    if first != second
    ]

metrics = {
    file_name : [
        [f"{task_prefix}{combination[0]}_{combination[1]}_eval_{suffix}" for suffix in metric_suffixes]
        for combination in combinations
    ] for file_name in file_names
}

model_names = file_names

# Function to extract values for a given metric
def extract_values(log_content, metric):
    pattern = rf'{metric} = (-?[\d.]+)'
    matches = re.findall(pattern, log_content)
    return [float(value) for value in matches]

# Function to extract condensed metric names
def condensed_metric_name(metric):
    if task_prefix.startswith('onestop'): 
        metric = metric.replace(task_prefix , '').replace('_eval_', '').replace('adv_', '').replace('int_', '').replace('ele_', '')
    else:
        metric = metric.replace(task_prefix , '').replace('_eval_', '').replace('Level0_', '').replace('Level1_', '').replace('Level3_', '')
    if metric.endswith("PERCENTAGE"):
        metric = metric[:-10] + 'PERC'
    if task_prefix.startswith('onestop'): 
        return metric[3:]
    else:
        return metric[6:]
    

# Define the metrics of interest
metrics_of_interest = ["SARI",
                       "LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE", 
                       "CORRECT_PARAPHRASE_PERCENTAGE", 
                       "SRC_PRED_EXACT_COPY",
                       "PRED_ONLY_GFI", 
                       "PRED_ONLY_FRE", 
                       "PRED_ONLY_FKGL", 
                       "PRED_ONLY_ARI", 
                       "PRED_ONLY_ASL"]

# Initialize a nested dictionary to store the top 10 metric values for each model
top_metric_values = {model[:-4]: {metric: [] for metric in metrics_of_interest} for model in model_names}

for index in range(6):
    top_SARI_scores = {}
    top_SARI_steps = {}
    # extract the SARI scores for each model and get the indices of top 10 scores
    for j, (log_content, model_name) in enumerate(zip(log_contents, model_names)):
        # Get all SARI metrics for this model
        SARI_metric = [metric for sublist in metrics[model_name] for metric in sublist if "SARI" in metric][index]
        print(SARI_metric)
        SARI_values = extract_values(log_content, SARI_metric)[:-4]
        
        # Enumerate SARI values to capture indices, then sort by SARI score in descending order
        sorted_scores = sorted(list(enumerate(SARI_values)), key=lambda x: x[1], reverse=True)[:25]
        
        # Extract the indices of top 10 scores
        top_indices = [x[0] for x in sorted_scores]
        
        # For each metric of interest, get the values corresponding to the top 10 SARI scores
        for metric in metrics_of_interest:
            full_metric = [m for sublist in metrics[model_name] for m in sublist if metric in m][index]
            print(full_metric)
            metric_values = extract_values(log_content, full_metric)[:-4]
            if metric == "LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE" or metric == "CORRECT_PARAPHRASE_PERCENTAGE" or metric == "SRC_PRED_EXACT_COPY":
                metric_values = [value/100 for value in metric_values]  
            if not metric_values:
                top_metric_values[model_name[:-4]][metric] = [0.555 for i in top_indices]
            else:
                top_metric_values[model_name[:-4]][metric] = [metric_values[i] for i in top_indices]

    # Now convert this nested dictionary to a DataFrame
    # First convert the inner dictionary to a DataFrame
    for model, metrics_dict in top_metric_values.items():
        top_metric_values[model] = pd.DataFrame(metrics_dict)

    # Now convert the outer dictionary to a DataFrame
    df_top_metrics = pd.concat(top_metric_values, axis=1)

    # Save to csv
    pair_name = '_'.join(SARI_metric.split('_')[3:5])
    df_top_metrics.to_csv(pair_name + '_top_metrics_values.csv')

