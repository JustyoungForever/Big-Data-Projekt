import os
import pandas as pd


raw_data_dir = '../raw_data'
problematic_datasets = []
cleaned_datasets = []

def is_valid_columns(columns):
    # Check if any column name is invalid (e.g., "Unnamed")
    return not any(col.startswith('Unnamed') for col in columns)

# Function to generate description
def generate_description(folder_name, file_name, df):
    rows, columns = df.shape
    column_names = ', '.join(df.columns)
    return f"{folder_name}'s {file_name}, containing {columns} columns: [{column_names}], with {rows} rows."

# traverse structure
for root, dirs, files in os.walk(raw_data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        file_name = file
        file_extension = os.path.splitext(file)[1].lower()
        relative_path = os.path.relpath(file_path, raw_data_dir)
        folder_name = os.path.basename(os.path.dirname(file_path))

        # Skip non-table files
        if file_extension not in ['.csv', '.xlsx', '.xls']:
            continue

        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, nrows=5)  # 5 rows
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=5)  #  5 rows
            
            if not is_valid_columns(df.columns):
                raise ValueError("Invalid column names detected")

            description = generate_description(folder_name, file_name, df)
            cleaned_datasets.append({
                'FileName': file_name,
                'FilePath': relative_path,
                'Format': file_extension,
                'Description': description
            })
        except Exception as e:
            problematic_datasets.append({
                'FileName': file_name,
                'FilePath': relative_path,
                'Error': str(e)
            })

cleaned_df = pd.DataFrame(cleaned_datasets)
cleaned_df.to_csv('dataset_overview_cleaned.csv', index=False, encoding='utf-8')

problematic_df = pd.DataFrame(problematic_datasets)
problematic_df.to_csv('problematic_datasets_processed.csv', index=False, encoding='utf-8')

print("Processing completed.")
print("Cleaned dataset overview saved to 'dataset_overview_cleaned.csv'.")
print("Problematic datasets saved to 'problematic_datasets_processed.csv'.")
