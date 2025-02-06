import os
import pandas as pd

overview_file_path = './dataset_overview_cleaned.csv'
raw_data_dir = '../raw_data'

keywords = ['AI', 'ML', 'automation', 'job', 'salary', 'industry', 'risk']

relevant_fields = ['JobTitle', 'Industry', 'Salary', 'Risk', 'Location', 'Year']
dynamic_column_mapping = {}

overview_df = pd.read_csv(overview_file_path)

# Create a dynamic column mapping 
for _, row in overview_df.iterrows():
    fields = row['Description'].split(':')[-1].strip().split(', ')
    for field in fields:
        if 'Job' in field:
            dynamic_column_mapping[field] = 'JobTitle'
        elif 'Industry' in field:
            dynamic_column_mapping[field] = 'Industry'
        elif 'Salary' in field or 'Income' in field:
            dynamic_column_mapping[field] = 'Salary'
        elif 'Risk' in field:
            dynamic_column_mapping[field] = 'Risk'
        elif 'Location' in field or 'Region' in field:
            dynamic_column_mapping[field] = 'Location'
        elif 'Year' in field or 'Date' in field:
            dynamic_column_mapping[field] = 'Year'

cleaning_logs = []
valid_datasets = []

# Filter 
filtered_overview = overview_df[
    overview_df['Description'].str.contains('|'.join(keywords), case=False, na=False)
]

# Process dataset
for _, row in filtered_overview.iterrows():
    file_name = row['FileName']
    file_path = os.path.join(raw_data_dir, row['FilePath'])
    file_format = row['Format'].lower()
    
    try:
        if file_format == '.csv':
            df = pd.read_csv(file_path)
        elif file_format in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            cleaning_logs.append(f"Skipped {file_name}: Unsupported file format.")
            continue

        # Rename
        df.rename(columns=dynamic_column_mapping, inplace=True)

        # Filter
        valid_columns = [col for col in df.columns if col in relevant_fields]
        if len(valid_columns) == 0:
            cleaning_logs.append(f"Skipped {file_name}: No relevant columns found after renaming.")
            continue
        
        df = df[valid_columns]

        # Add a column indicating the source dataset
        df['SourceDataset'] = file_name

        # Reset index and check for duplicates
        df.reset_index(drop=True, inplace=True)

        # Ensure no duplicate indices
        if not df.index.is_unique:
            cleaning_logs.append(f"Skipped {file_name}: Duplicate indices detected.")
            continue

        if df.empty:
            cleaning_logs.append(f"Skipped {file_name}: Empty DataFrame after processing.")
            continue

        valid_datasets.append(df)
        cleaning_logs.append(f"Processed {file_name} successfully with {len(df)} rows and {len(df.columns)} columns.")

    except Exception as e:
        cleaning_logs.append(f"Failed to process {file_name}: {e}")

# Merge 
if valid_datasets:
    for i, df in enumerate(valid_datasets):
        df = df.loc[:, ~df.columns.duplicated()]
        
        df = df.reindex(columns=relevant_fields + ['SourceDataset'], fill_value=None)
        valid_datasets[i] = df

    master_df = pd.concat(valid_datasets, ignore_index=True)

    master_df.to_csv('cleaned_ai_job_datasets_dynamic_with_source.csv', index=False, encoding='utf-8')
    print("Cleaned and merged dataset saved to 'cleaned_ai_job_datasets_dynamic_with_source.csv'.")
else:
    print("No valid datasets found after filtering and cleaning.")

with open('cleaning_logs_dynamic_with_source.txt', 'w') as log_file:
    for log in cleaning_logs:
        log_file.write(log + '\n')
    print("Cleaning logs saved to 'cleaning_logs_dynamic_with_source.txt'.")
