import pandas as pd
import shutil
import os

def copy_files_with_chromosome_count(xlsx_path, img_path, output_dir):
    # Read the Excel file
    df = pd.read_excel(xlsx_path, dtype=str)  # Treat all values as strings to avoid type issues
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the 'chromosomeCount' column exists
    if 'chromosomeCount' not in df.columns:
        print("Error: 'chromosomeCount' column not found in the spreadsheet.")
        return

    # Iterate over rows where chromosomeCount is not empty
    for _, row in df.iterrows():
        if pd.notna(row['chromosomeCount']) and str(row['chromosomeCount']).strip():
            file_path = os.path.join(img_path, row.get('catalogNumber') + '.jpg')  # Adjust the column name if needed
            if file_path and os.path.exists(file_path):
                shutil.copy(file_path, output_dir)
                print(f"Copied: {file_path} -> {output_dir}")
            else:
                print(f"File not found: {file_path}")

# Example usage
xlsx_path = "D:/T_Downloads/shelly_galax_gbif/gbif_2025_02_07__16-31-34/Transcription/transcribed.xlsx"
img_path = "D:/T_Downloads/shelly_galax_gbif/Galax/img"
output_dir = "D:/T_Downloads/shelly_galax_gbif/gbif_2025_02_07__16-31-34/Transcription/img_subset"
copy_files_with_chromosome_count(xlsx_path, img_path, output_dir)
