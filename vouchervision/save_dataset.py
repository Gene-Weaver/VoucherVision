from datasets import load_dataset

# Load the dataset
dataset = load_dataset("phyloforfun/HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05")

# Define the directory where you want to save the files
save_dir = "D:/Dropbox/VoucherVision/datasets/SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05"

# Save each split as a JSONL file in the specified directory
for split, split_dataset in dataset.items():
    split_dataset.to_json(f"{save_dir}/SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05-{split}.jsonl")


'''import json # convert to google

# Load the JSONL file
input_file_path = '/mnt/data/SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05-train.jsonl'
output_file_path = '/mnt/data/SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05-train-converted.jsonl'

# Define the conversion function
def convert_record(record):
    return {
        "input_text": record.get('instruction', '') + ' ' + record.get('input', ''),
        "target_text": record.get('output', '')
    }

# Convert and save the new JSONL file
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        record = json.loads(line)
        converted_record = convert_record(record)
        outfile.write(json.dumps(converted_record) + '\n')

output_file_path'''
