import re
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the content of the log files
file_names = [
    'small_separate_random.log'
]

task_prefix = 'onestop_parallel_sentence_'
log_contents = []
for file_name in file_names:
    with open(file_name, 'r') as file:
        log_contents.append(file.read())


new_metric_suffixes = [
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
    "LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE",
    "CORRECT_PARAPHRASE_PERCENTAGE",
    "AVG_PRED_TGT_LEV_DIST",
    "AVG_FKGL_SRC_PRED_DIFF",
    "AVG_FKGL_PRED_TGT_ABSDIFF",
    "PRED_ONLY_GFI",
    "PRED_ONLY_FKGL",
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

combinations = [
    (first, second)
    for first in ['adv', 'int', 'ele']
    for second in ['adv', 'int', 'ele']
    if first != second
]
model_names = [file[:-4] for file in file_names]
metrics = {
    model : [
        [f"{task_prefix}{combination[0]}_{combination[1]}_eval_{suffix}" for suffix in metric_suffixes]
        for combination in combinations
    ]
    for model in model_names
}

# Function to extract values for a given metric
def extract_values(log_content, metric):
    pattern = rf'{metric} = (-?[\d.]+)'
    matches = re.findall(pattern, log_content)
    return [float(value) for value in matches]

# Function to extract condensed metric names
def condensed_metric_name(metric):
    metric = metric.replace(task_prefix , '').replace('_eval_', '').replace('adv_', '').replace('int_', '').replace('ele_', '')
    if metric.endswith("PERCENTAGE"):
        metric = metric[:-10] + 'PERC'
    return metric[3:]


# Create directories for the two types of plots and metric sets
for folder_name in ['min_max_plots', '5_95_percentile_plots']:
    for metric_set in ['primary', 'secondary']:
        if not os.path.exists(f'{folder_name}/{metric_set}'):
            os.makedirs(f'{folder_name}/{metric_set}')

colors = ['b', 'r']
            
# Plot the metrics independently for all models
for metric_id, metric in enumerate(new_metric_suffixes):
    # Calculate the global minimum and maximum values
    global_min = +10000
    global_max = -10000
    # Create a list to hold all values
    all_values = []

    # Go through each combination and each model once to find the global min and max
    for i, combination in enumerate(combinations):
        for j, (log_content, model_name) in enumerate(zip(log_contents, model_names)):
            current_metric = metrics[model_name][i][metric_id]
            values = extract_values(log_content, current_metric)[:-4]
            if not values:
                print(f"No values found for metric: {current_metric}")
            else:
                all_values.extend(values)
                global_min = min(global_min, min(values))
                global_max = max(global_max, max(values))

    # Calculate the 5th and 95th percentiles
    percentile_min = np.percentile(all_values, 5)
    percentile_max = np.percentile(all_values, 95)

    for ylim_values, folder_name in [((global_min, global_max), 'min_max_plots'), ((percentile_min, percentile_max), '5_95_percentile_plots')]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 12), sharex=True, sharey=True)
        for i, combination in enumerate(combinations):
            for j, (log_content, model_name) in enumerate(zip(log_contents, model_names)):
                current_metric = metrics[model_name][i][metric_id]
                values = extract_values(log_content, current_metric)[:-4]
                row, col = divmod(i, 3)
                steps = np.linspace(0, 65536, num=len(values))

                axes[row, col].plot(steps, values, color=colors[j], label=model_name)  # specify color and label
                axes[row, col].set_xlabel('Steps')
                axes[row, col].set_ylabel(condensed_metric_name(current_metric))
                axes[row, col].set_title(f"{combination[0]}_{combination[1]}: {condensed_metric_name(current_metric)}")
                axes[row, col].set_ylim(ylim_values)
                axes[row, col].set_xticks(np.arange(0, 65536, 15000))
                axes[row, col].legend()

        plt.tight_layout()

        if metric in primary_metrics:
            plt.savefig(f'{folder_name}/primary/{metric}_comparison_plot.png', dpi=300)
        else:
            plt.savefig(f'{folder_name}/secondary/{metric}_comparison_plot.png', dpi=300)

        plt.close(fig)