import json


def validate_and_align_JSON_keys_with_template(data, JSON_dict_structure):
    list_of_nulls = ['unknown','not provided', 'missing', 'na', 'none', 'n/a', 'null', 'unspecified', 'not known','not given',
                                     'TBD', 'tbd',
                                    'not provided in the text', 'not found in the text', 'Not found in OCR text', 'not found in ocr text',
                                    'not in the text', 'not provided', 'not found',
                                    'not provided in the ocr', 'not found in the ocr', 
                                    'not in the ocr', 
                                    'not provided in the ocr text', 'not found in the ocr text', 
                                    "not specified in the given text.",
                                    "not specified in the given text",
                                    "not specified in the text.",
                                    "not specified in the text",
                                    "not specified in text.",
                                    "not specified in text",
                                    "not specified in ocr",
                                    "not specified",
                                    "google handwritten ocr",
                                    "google printed ocr",
                                    "gpt-4o-mini ocr",
                                    "qwen2-vl-7b-instruct ocr",
                                    "qwen2-vl-72b-instruct ocr",
                                    "llama-3.2-90b-vision-instruct ocr",
                                    'not in the ocr text', 
                                    'Not provided in ocr text',
                                    'not provided in ocr text',
                                    'n/a n/a','n/a, n/a','Not applicable','not applicable',
                                    'n/a, n/a, n/a','n/a n/a, n/a','n/a, n/a n/a','n/a n/a n/a',
                                    'n/a, n/a, n/a, n/a','n/a n/a n/a n/a','n/a n/a, n/a, n/a','n/a, n/a n/a, n/a','n/a, n/a, n/a n/a',
                                    'n/a n/a n/a, n/a','n/a, n/a n/a n/a',
                                    'n/a n/a, n/a n/a',]
    list_of_ocr_nulls = [
            'Pixtral-12B-2409 OCR',
            'Qwen2-VL-7B-Instruct OCR',
            'Qwen2-VL-72B-Instruct OCR',
            'Llama-3.2-90B-Vision-Instruct OCR',
            'Qwen2.5-72B-Instruct OCR',
            'Qwen2.5-Coder-32B-Instruct OCR',
            'Llama-3.2-3B-Instruct OCR',
            'Meta-Llama-3.1-405B-Instruct OCR',
            'Meta-Llama-3.1-405B-FP8 OCR',
            'Meta-Llama-3.1-8B-Instruct OCR',
            'Meta-Llama-3.1-70B-Instruct OCR',
            'Meta-Llama-3-70B-Instruct OCR',
            'Hermes-3-Llama-3.1-70B OCR',
            'DeepSeek-V2.5 OCR',
            'Gemini-1.5-Pro OCR',
            'Gemini OCR',
            'Pixtral-12B-2409',
            'Qwen2-VL-7B-Instruct',
            'Qwen2-VL-72B-Instruct',
            'Llama-3.2-90B-Vision-Instruct',
            'Qwen2.5-72B-Instruct',
            'Qwen2.5-Coder-32B-Instruct',
            'Llama-3.2-3B-Instruct',
            'Meta-Llama-3.1-405B-Instruct',
            'Meta-Llama-3.1-405B-FP8',
            'Meta-Llama-3.1-8B-Instruct',
            'Meta-Llama-3.1-70B-Instruct',
            'Meta-Llama-3-70B-Instruct',
            'Hermes-3-Llama-3.1-70B',
            'DeepSeek-V2.5',
            'Gemini-1.5-Pro',
            'Gemini',
            ]
    list_of_ocr_nulls_lower = [item.lower() for item in list_of_ocr_nulls]
    list_of_nulls_lower = [item.lower() for item in list_of_nulls]
    data = validate_JSON(data)
    if data is None:
        return None
    else:
        # Handle duplicate fields by creating a case-insensitive dictionary for tracking duplicates
        # and keeping the one with the most content
        cleaned_data = {}
        field_content_length = {}  # Track content length for each field (case-insensitive)
        
        # First pass: Track all fields and their content lengths
        for key, value in data.items():
            key_lower = key.lower()
            
            # Convert value to string for length comparison
            if value is None:
                str_value = ''
            elif isinstance(value, list):
                str_value = ', '.join(str(item) for item in value)
            else:
                str_value = str(value)
                
            # Check if value is in nulls list
            if str_value.lower() in list_of_ocr_nulls_lower or str_value.lower() in list_of_nulls_lower:
                str_value = ''
                
            # Track the length of content for this field
            current_length = len(str_value)
            
            # If we've seen this field before, compare content length
            if key_lower in field_content_length:
                if current_length > field_content_length[key_lower][1]:
                    # This instance has more content, update our tracking
                    field_content_length[key_lower] = (key, current_length, str_value)
            else:
                # First time seeing this field
                field_content_length[key_lower] = (key, current_length, str_value)
                
        # Second pass: Use only the fields with the most content
        for key_lower, (original_key, length, value) in field_content_length.items():
            cleaned_data[original_key] = value.replace('*', '') if value else ''
                
        ### align the keys with the template
        # Create a new dictionary with the same order of keys as JSON_dict_structure
        ordered_data = {}

        # This will catch cases where the LLM JSON case does not match the required JSON key's case
        for key in JSON_dict_structure:
            truth_key_lower = key.lower()
            
            # Look for the key in a case-insensitive way
            llm_value = ''
            if key in cleaned_data:
                llm_value = cleaned_data[key]
            else:
                # Try to find a case-insensitive match
                for data_key in cleaned_data:
                    if data_key.lower() == truth_key_lower:
                        llm_value = cleaned_data[data_key]
                        break
            
            # Copy the value from cleaned_data if it exists, else use an empty string
            ordered_data[key] = llm_value

        return ordered_data




def validate_JSON(data):
    if isinstance(data, dict):
        return data
    else:
        if isinstance(data, list):
            data = data[0]
        try:
            json_candidate = json.loads(data) # decoding the JSON data
            if isinstance(json_candidate, list):
                json_candidate = json_candidate[0]
            
            if isinstance(json_candidate, dict):
                data = json_candidate
                return data
            else:
                return None
        except:
            return None  
        



#### A manual method for pulling a JSON object out of a verbose LLM response.
#### It's messy butr works well enough. Only use if JSONparsing is not available
def extract_json_dict_manual(text):
    text = text.strip().replace("\n", "").replace("\\\\", "")
    # Find the first opening curly brace
    start_index = text.find('{')
    # Find the last closing curly brace
    end_index = text.rfind('}') + 1

    text = text[start_index:end_index]

    # Find the first opening curly brace
    start_index = text.find('{')
    # Find the last closing curly brace
    end_index = text.rfind('}') + 1

    # Check and remove backslash immediately after the opening curly brace
    if text[start_index + 1] == "\\":
        text = text[:start_index + 1] + text[start_index + 2:]

    # Find the first opening curly brace
    start_index = text.find('{')
    # Find the last closing curly brace
    end_index = text.rfind('}') + 1

    # print(text[end_index-2])
    if text[end_index-2] == "\\":
        text = text[:end_index-2] + text[end_index-1]
    else:
        text = text

    # Find the first opening curly brace
    start_index = text.find('{')
    # Find the last closing curly brace
    end_index = text.rfind('}') + 1

    # print(text[end_index-2])
    if text[end_index-2] == "\\":
        text = text[:end_index-3] + text[end_index-1]
    else:
        text = text

    # Trim whitespace
    json_str = text

    # Print the JSON string for inspection
    # print("Extracted JSON string:", json_str)

    # Convert JSON string to Python dictionary
    try:
        # If necessary, replace newline characters
        # json_str = json_str.replace('\n', '\\n')

        json_dict = json.loads(json_str)
        return json_dict
    except Exception as e:
        print("Error parsing JSON:", e)
        return None
'''
if __name__ == "__main__":
    tex = """Extracted JSON string: {"catalogNumber": "MPU395640",
    "order": "Monimizeae",
    "family": "Monimiaceae",
    "scientificName": "Hedycarya parvifolia",
    "scientificNameAuthorship": "Perkins & Schltr.",
    "genus": "Hedycarya",
    "subgenus": null,
    "specificEpithet": "parvifolia",
    "infraspecificEpithet": null,
    "identifiedBy": null,
    "recordedBy": "R. Pouteau & J. Munzinger",
    "recordNumber": "RP 1",
    "verbatimEventDate": "26-2-2013",
    "eventDate": "2013-02-26",
    "habitat": "Ultramafique Long. 165 ° 52'21  E, Lat. 21 ° 29'19  S, Maquis",
    "occurrenceRemarks": "Fruit Arbuste, Ht. 1,5 mètre (s), Fruits verts (immatures) à noirs (matures), Coll. R. Pouteau & J. Munzinger N° RP 1, Dupl. P - MPU, RECOLNAT, Herbier IRD de Nouvelle - Calédonie Poutan 1, Golden Thread, Alt. 818 mètre, Diam. 2 cm, Date 26-2-2013",
    "country": "Nouvelle Calédonie",
    "stateProvince": null,
    "county": null,
    "municipality": null,
    "locality": "Antenne du plateau de Bouakaine",
    "degreeOfEstablishment": "cultivated",
    "decimalLatitude": -21.488611,
    "decimalLongitude": 165.8725,
    "verbatimCoordinates": "Long. 165 ° 52'21  E, Lat. 21 ° 29'19  S",
    "minimumElevationInMeters": 818,
    "maximumElevationInMeters": 818
    \\}"""
    new = extract_json_dict(tex)
'''
