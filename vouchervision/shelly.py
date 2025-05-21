import pandas as pd
import shutil
import os

def copy_files_with_chromosome_count(xlsx_path, img_path, output_dir):
    # Read the Excel file
    df = pd.read_excel(xlsx_path, dtype=str)  # Treat all values as strings to avoid type issues
    filtered_df = df.dropna(subset=['chromosomeCount', 'guardCell'], how='all')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the 'chromosomeCount' column exists
    if 'chromosomeCount' not in df.columns:
        print("Error: 'chromosomeCount' column not found in the spreadsheet.")
        return

    # Iterate over rows where chromosomeCount is not empty
    for _, row in filtered_df.iterrows():
        if pd.notna(row['chromosomeCount']) and str(row['chromosomeCount']).strip():
            file_path = os.path.join(img_path, row.get('catalogNumber') + '.jpg')  # Adjust the column name if needed
            if file_path and os.path.exists(file_path):
                shutil.copy2(file_path, output_dir)
                print(f"Copied: {file_path} -> {output_dir}")
            else:
                print(f"File not found: {file_path}")

# Usage
# xlsx_path = "D:/T_Downloads/shelly_galax_gbif/gbif_2025_02_07__16-31-34/Transcription/transcribed.xlsx"
# img_path = "D:/T_Downloads/shelly_galax_gbif/Galax/img"
# output_dir = "D:/T_Downloads/shelly_galax_gbif/gbif_2025_02_07__16-31-34/Transcription/img_subset"

xlsx_path = "D:/D_Desktop/2025-02-13_MichelleGaynor_out/bothGeminiOCR_new_prompt_2025_02_25__16-49-43/Transcription/transcribed.xlsx"
img_path = "D:/D_Desktop/2025-02-13_MichelleGaynor_out/bothGeminiOCR_new_prompt/Cropped_Images/By_Class/label"
output_dir = "D:/D_Desktop/2025-02-13_MichelleGaynor_with_chromosomes"
copy_files_with_chromosome_count(xlsx_path, img_path, output_dir)
