import requests
from bs4 import BeautifulSoup
import yaml
import os
from datetime import datetime
from model_maps import ModelMaps

def format_float(value):
    return f"{value:.6f}".rstrip('0').rstrip('.') + ('0' if '.' not in f"{value:.1f}" else '')

def update_api_costs(test_mode=True):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory
    parent_dir = os.path.dirname(current_dir)

    # Load existing api_cost.yaml
    api_cost_path = os.path.join(parent_dir, 'api_cost', 'api_cost.yaml')
    with open(api_cost_path, 'r') as file:
        api_costs_raw = file.read()
    api_costs = yaml.safe_load(api_costs_raw)

    # Get all model names and their API names
    model_maps = ModelMaps()
    models_to_check = {}
    for key in model_maps.get_all_mapping_cost().values():
        try:
            api_name = model_maps.get_API_name(key)
            models_to_check[key] = api_name
        except ValueError:
            pass  # Skip models that don't have an API name

    # Aliases for website models
    website_aliases = {
        # Mistral Models
        'mistral-large': 'MISTRAL_LARGE',
        'mistral-medium': 'MISTRAL_MEDIUM',
        'mistral-small': 'MISTRAL_SMALL',
        'mixtral-8x7b': 'OPEN_MIXTRAL_8X7B',
        'mistral-7b': 'OPEN_MISTRAL_7B',
        
        # Google Gemini Models
        'gemini-pro': 'GEMINI_PRO',
        'gemini-1.5-pro': 'GEMINI_1_5_PRO',
        'gemini-flash-1.5': 'GEMINI_1_5_FLASH',
        
        # OpenAI GPT-4 and GPT-3.5 Models
        'gpt-4': 'GPT_4',
        'gpt-4-32k': 'GPT_4_32K',
        'gpt-4-0125-preview': 'GPT_4_TURBO_0125',
        'gpt-4-1106-preview': 'GPT_4_TURBO_1106',
        'gpt-4-turbo-2024-04-09': 'GPT_4_TURBO_2024_04_09',
        'gpt-3.5-turbo-0125': 'GPT_3_5',
        'gpt-3.5-turbo-instruct': 'GPT_3_5_INSTRUCT',
        
        # OpenAI GPT-4o Variants
        'gpt-4o-2024-08-06': 'GPT_4o_2024_08_06',
        'gpt-4o-2024-05-13': 'GPT_4o_2024_05_13',
        'gpt-4o-mini': 'GPT_4o_mini_2024_07_18',

        # OpenAI GPT-4o Variants
        'gpt-4-32k': 'AZURE_GPT_4_32K',
        'gpt-4-0125-preview': 'AZURE_GPT_4_TURBO_0125',
        'gpt-4-1106-preview': 'AZURE_GPT_4_TURBO_1106',
        'gpt-3.5-turbo-instruct': 'AZURE_GPT_3_5_INSTRUCT',
        'gpt-3.5-turbo-0125': 'AZURE_GPT_3_5',
    }


    # Fetch the webpage
    url = 'https://llmpricecheck.com/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table
    table = soup.find('table')

    # Keep track of updated models
    updated_models = set()

    # Extract data from the table
    for row in table.find_all('tr')[1:]:  # Skip header row
        columns = row.find_all('td')
        if len(columns) >= 6:
            model = columns[0].text.strip().lower()
            provider = columns[1].text.strip().lower()
            input_price = float(columns[4].text.strip().replace('$', ''))
            output_price = float(columns[5].text.strip().replace('$', ''))

            # Only use prices from OpenAI, MistralAI, and Google as the provider
            if provider not in ['openai','openrouter', 'mistral', 'google']:
                print(f"{provider} not supported")
                continue

            # Check if this model is in our aliases, and allow matching for gpt-4o-mini
            if model in website_aliases:
                key = website_aliases[model]
                if key in api_costs:
                    print(api_costs[key]['in'])
                    if api_costs[key]['in'] != input_price or api_costs[key]['out'] != output_price:
                        api_costs[key]['in'] = input_price
                        api_costs[key]['out'] = output_price
                        updated_models.add(key)
                else:
                    print(f"Model {model} not found in api_costs")
            elif 'gpt-4o-mini' in model and 'gpt-4o-mini' in website_aliases:
                key = website_aliases['gpt-4o-mini']
                if key in api_costs:
                    if api_costs[key]['in'] != input_price or api_costs[key]['out'] != output_price:
                        api_costs[key]['in'] = input_price
                        api_costs[key]['out'] = output_price
                        updated_models.add(key)
                else:
                    print(f"Model {model} not found in api_costs")
            else:
                print(f"Model {model} not found in aliases")

    # Debugging: Print out the fetched prices
    print("Fetched prices from the website:")
    for key, value in api_costs.items():
        if key in updated_models:
            print(f"Updated {key}: in = {value['in']}, out = {value['out']}")

    # Update the YAML content with comments
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_content = []
    current_model = None
    last_updated_comment = None

    for line in api_costs_raw.split('\n'):
        if line.strip().startswith('# UPDATED:'):
            last_updated_comment = line
        elif line.strip() and not line.strip().startswith('#'):
            if ':' in line and not line.strip().startswith('in:') and not line.strip().startswith('out:'):
                current_model = line.split(':')[0].strip()
                if current_model in updated_models:
                    updated_content.append(f"# UPDATED: {update_time}")
                    updated_content.append(f"{current_model}:")
                    updated_content.append(f"  in: {format_float(api_costs[current_model]['in'])}")
                    updated_content.append(f"  out: {format_float(api_costs[current_model]['out'])}")
                else:
                    if last_updated_comment:
                        updated_content.append(last_updated_comment)
                    updated_content.append(line)
                last_updated_comment = None
            elif current_model not in updated_models:
                updated_content.append(line)
        else:
            updated_content.append(line)

    updated_content = '\n'.join(updated_content)

    if test_mode:
        # Create a test file in the same location as the original file
        test_file_path = api_cost_path.replace('.yaml', '_test.yaml')
        with open(test_file_path, 'w') as file:
            file.write(updated_content)
        print(f"Test mode: Updated API costs have been written to: {test_file_path}")
        return test_file_path
    else:
        # Write the updated api_costs back to the original YAML file
        with open(api_cost_path, 'w') as file:
            file.write(updated_content)
        print("API costs have been updated in api_cost.yaml")
        return api_cost_path

if __name__ == "__main__":
    update_api_costs(test_mode=False)
