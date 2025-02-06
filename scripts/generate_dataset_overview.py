import os
import pandas as pd


raw_data_dir = '../raw_data'
problematic_files_path = './problematic_datasets_processed.csv'  
problematic_files_df = pd.read_csv(problematic_files_path)
problematic_file_paths = set(problematic_files_df['FilePath'])

dataset_info = []

# Function to generate description 
def generate_description(folder_name, file_name, df):
    try:
        rows, columns = df.shape
        column_names = ', '.join(df.columns)
        return f"{folder_name}'s {file_name}, containing {columns} columns: [{column_names}], with {rows} rows."
    except Exception as e:
        return f"Failed to generate description for {folder_name}'s {file_name}, error: {e}"

# Traverse the folder structure
for root, dirs, files in os.walk(raw_data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        file_name = file
        file_extension = os.path.splitext(file)[1].lower()
        relative_path = os.path.relpath(file_path, raw_data_dir)
        folder_name = os.path.basename(os.path.dirname(file_path))

        # Skip non-table files
        if file_extension not in ['.csv', '.xlsx', '.xls'] or relative_path in problematic_file_paths:
            continue

        description = ""
        rows, columns = 0, 0
        column_names = ""

        try:

            if file_extension == '.csv':
                df = pd.read_csv(file_path, nrows=5)  
                description = generate_description(folder_name, file_name, df)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=5)  
                description = generate_description(folder_name, file_name, df)
        except Exception as e:

            continue

        # Append to the list
        dataset_info.append({
            'FileName': file_name,
            'FilePath': relative_path,
            'Format': file_extension if file_extension else 'Unknown',
            'Description': description
        })


df_info = pd.DataFrame(dataset_info)
df_info.to_csv('dataset_overview_cleaned.csv', index=False, encoding='utf-8')

print("Cleaned dataset overview has been generated and saved to 'dataset_overview_cleaned.csv'.")
