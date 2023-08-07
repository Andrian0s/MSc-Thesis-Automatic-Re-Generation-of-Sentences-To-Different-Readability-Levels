import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('ele_adv_top_metrics_values_edited.csv', header=None)

# Initialize an empty dataframe to store the final output
output = pd.DataFrame(columns=['Model', 'Metric', 'Value'])

# Mapping dictionary for model names
model_mapping = {
    'small_pairwise_adapters': 'Small Pairwise Adapters',
    'base_pairwise_adapters': 'Base Pairwise Adapters',
    'small_pairwise_full_model': 'Small Pairwise Full Model',
    'base_pairwise_full_model': 'Base Pairwise Full Model',
    'small_string_prefix': 'Small String Prefix',
    'base_string_prefix': 'Base String Prefix',
    'small_combined_readability_init': 'Small H++(Combined)',
    'base_combined_readability_init': 'Base H++(Combined)',
    'small_separate_readability_init': 'Small H++(Separate)',
    'base_separate_readability_init': 'Base H++(Separate)',
}

# Mapping dictionary for metrics
metric_mapping = {
    'CORRECT_PARAPHRASE_PERCENTAGE' : 'Corr. P',
    'LINGUISTIC_SENTENCE_ACCEPTABILITY_PERCENTAGE' : 'Fluency',
    'SRC_PRED_EXACT_COPY' : 'Copies',
    'SARI' : 'SARI',
    'PRED_ONLY_GFI' : 'GFI',
    'PRED_ONLY_FRE' : 'FRE',
    'PRED_ONLY_FKGL' : 'FKGL',
    'PRED_ONLY_ARI' : 'ARI',
    'PRED_ONLY_ASL' : 'ASL'
}

# Iterate over each group of 9 columns
for i in range(0, df.shape[1], 9):
    # Extract the model, metrics, and values
    model = df.iloc[0, i]
    metrics = df.iloc[1, i:i+9]
    values = df.iloc[2:, i:i+9]
    
    # Flatten the values and repeat the model and metrics to align with values
    model_col = pd.Series([model] * len(metrics) * len(values))
    metric_col = pd.Series(metrics.tolist() * len(values))
    value_col = values.values.flatten()

    # Combine the model, metrics, and values into a dataframe
    df_temp = pd.DataFrame({'Model': model_col, 'Metric': metric_col, 'Value': value_col})
    
    # Append this dataframe to the output dataframe
    output = pd.concat([output, df_temp], ignore_index=True)

# Replace the model names according to the mapping
output['Model'] = output['Model'].replace(model_mapping)

# Replace the metric names according to the mapping
output['Metric'] = output['Metric'].replace(metric_mapping)

# Save the output dataframe to a new CSV file
output.to_csv('ele_adv_top_metrics_values_reshaped.csv', index=False)
