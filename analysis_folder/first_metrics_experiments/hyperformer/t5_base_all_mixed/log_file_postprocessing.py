import re
import matplotlib.pyplot as plt

# Read the content of the log files
file_name = 't5_base_hyperformer.log'
file_names = [
    file_name
]
log_contents = []
for file_name in file_names:
    with open(file_name, 'r') as file:
        log_contents.append(file.read())

# Define the dictionary of metrics to extract and visualize
metric_suffixes = [
    'ARI', 'ASL', 'DCRF', 'FKGL', 'FRE', 'GFI', 'SMOG', 'loss', 'rouge1', 'rouge2', 'rougeL'
]

combinations = [
    (first, second)
    for first in ['adv', 'int', 'ele']
    for second in ['adv', 'int', 'ele']
    if first != second
]

metrics = {
    file_name : [
        [f"onestop_parallel_{combination[0]}_{combination[1]}_eval_{suffix}" for suffix in metric_suffixes]
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
    return metric.split('_')[-1]

# Plot the metrics independently for all models
for metric_id, metric in enumerate(metric_suffixes):
    fig, axes = plt.subplots(2, 3, figsize=(15, 12), sharex=True, sharey=True)
    
    for i, combination in enumerate(combinations):
        for j, (log_content, model_name) in enumerate(zip(log_contents, file_names)):
            current_metric = metrics[model_name][i][metric_id]
            values = extract_values(log_content, current_metric)[:-1]
            row, col = divmod(i, 3)
            axes[row, col].plot(values)
            axes[row, col].set_xlabel('Index')
            axes[row, col].set_ylabel(condensed_metric_name(current_metric))
            axes[row, col].set_title(f"{combination[0]}_{combination[1]}: {condensed_metric_name(current_metric)}")
    
    plt.tight_layout()
    plt.savefig(f'{condensed_metric_name(metric)}_comparison_plot.png', dpi=300)
