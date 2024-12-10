import pandas as pd

def clean_cataloged_date_column(path_template_csv, path_input_xlsx, output_path):
    # Read the template CSV to get header order and alias mappings
    template_df = pd.read_csv(path_template_csv, header=None)
    desired_headers = template_df.iloc[0].tolist()
    input_headers = template_df.iloc[1].tolist()

    # Create a mapping of input headers to desired headers
    header_mapping = dict(zip(input_headers, desired_headers))

    # Read the input Excel file
    input_df = pd.read_excel(path_input_xlsx)

    # Rename the columns based on the alias mapping, ignoring NaN mappings
    input_df.rename(columns={old: new for old, new in header_mapping.items() if pd.notna(old)}, inplace=True)

    # Add any missing columns from the template with empty values
    for col in desired_headers:
        if col not in input_df.columns:
            input_df[col] = ''

    # Reorder the columns to match the desired order
    input_df = input_df[desired_headers]

    # Clean 'Cataloged Date' column if it exists
    if 'Cataloged Date' in input_df.columns:
        input_df['Cataloged Date'] = input_df['Cataloged Date'].astype(str).apply(
            lambda x: x.split('T')[0].replace('_', '-') if pd.notna(x) else x
        )

    # Save the rearranged and cleaned DataFrame to a new Excel file
    input_df.to_excel(output_path, index=False)

    return output_path

if __name__ == '__main__':
    path_template_csv = 'S:/VoucherVision/Tools/Work_Bench_Headers_AllAsia_min.csv'
    path_input_xlsx = 'S:/VoucherVision/Transcription/willwe/Collage_HRT_test/Transcription/transcribed__edited__2024_12_09T09_57_28.xlsx'
    output_path = 'S:/VoucherVision/Transcription/willwe/Collage_HRT_test/Transcription/transcribed__final__2024_12_09T09_57_28.xlsx'

    cleaned_output_path = clean_cataloged_date_column(path_template_csv, path_input_xlsx, output_path)
    print(f"Cleaned and rearranged file saved at: {cleaned_output_path}")