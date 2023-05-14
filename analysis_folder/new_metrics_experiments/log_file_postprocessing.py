import re
import matplotlib.pyplot as plt
import numpy as np

# Read the content of the log files
file_name = 'output.log'
file_names = [
    file_name
]

task_prefix = 'onestop_parallel_text_'
log_contents = []
for file_name in file_names:
    with open(file_name, 'r') as file:
        log_contents.append(file.read())

# Define the dictionary of metrics to extract and visualize
metric_suffixes = [
    'ARI', 'ASL', 'DCRF', 'FKGL', 'FRE', 'GFI', 'SMOG', 'loss', 'rouge1', 'rouge2', 'rougeL'
]

new_metric_suffixes = [
    'AVG_ARI_SRC_PRED_DIFF', 
    'AVG_ASL_SRC_PRED_DIFF', 
    'AVG_DCRF_SRC_PRED_DIFF', 
    'AVG_FKGL_SRC_PRED_DIFF', 
    'AVG_FRE_SRC_PRED_DIFF', 
    'AVG_GFI_SRC_PRED_DIFF', 
    'AVG_SMOG_SRC_PRED_DIFF', 
    'loss', 
    'rouge1', 
    'rouge2', 
    'rougeL',
    'AVG_GFI_PRED_TGT_ABSDIFF',
    'AVG_FRE_PRED_TGT_ABSDIFF',
    'AVG_FKGL_PRED_TGT_ABSDIFF',
    'AVG_ARI_PRED_TGT_ABSDIFF',
    'AVG_DCRF_PRED_TGT_ABSDIFF',
    'AVG_SMOG_PRED_TGT_ABSDIFF',
    'AVG_ASL_PRED_TGT_ABSDIFF',
]

metric_suffixes = new_metric_suffixes


combinations = [
    (first, second)
    for first in ['adv', 'int', 'ele']
    for second in ['adv', 'int', 'ele']
    if first != second
]

metrics = {
    file_name : [
        [f"{task_prefix}{combination[0]}_{combination[1]}_eval_{suffix}" for suffix in metric_suffixes]
        for combination in combinations
    ]
}

model_names = [
    file_name 
]

# Function to extract values for a given metric
def extract_values(log_content, metric):
    pattern = rf'{metric} = ([\d.]+)'
    matches = re.findall(pattern, log_content)
    return [float(value) for value in matches]

# Function to extract condensed metric names
def condensed_metric_name(metric):
    metric = metric.replace(task_prefix , '').replace('_eval_', '').replace('adv_', '').replace('int_', '').replace('ele_', '')
    return metric[3:]

# Plot the metrics independently for all models
for metric_id, metric in enumerate(metric_suffixes):
    fig, axes = plt.subplots(2, 3, figsize=(15, 12), sharex=True, sharey=True)
    
    for i, combination in enumerate(combinations):
        for j, (log_content, model_name) in enumerate(zip(log_contents, file_names)):
            current_metric = metrics[model_name][i][metric_id]
            values = extract_values(log_content, current_metric)[:-1]
            row, col = divmod(i, 3)

            # Calculate the number of steps between each plot point
            steps = np.linspace(0, 65536, num=len(values))

            axes[row, col].plot(steps, values)
            axes[row, col].set_xlabel('Steps')
            axes[row, col].set_ylabel(condensed_metric_name(current_metric))
            axes[row, col].set_title(f"{combination[0]}_{combination[1]}: {condensed_metric_name(current_metric)}")

            # Set x-axis to show equal increments up to 65000
            axes[row, col].set_xticks(np.arange(0, 65536, 15000))  # Modify the step size according to your preference
    
    plt.tight_layout()
    plt.savefig(f'{metric}_comparison_plot.png', dpi=300)
