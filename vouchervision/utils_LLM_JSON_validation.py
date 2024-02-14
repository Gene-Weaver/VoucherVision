import json


def validate_and_align_JSON_keys_with_template(data, JSON_dict_structure):
    data = validate_JSON(data)
    if data is None:
        return None
    else:
        # Make sure that there are no literal list [] objects in the dict
        for key, value in data.items():
            if value is None:
                data[key] = ''
            elif isinstance(value, str):
                if value.lower() in ['unknown', 'not provided', 'missing', 'na', 'none', 'n/a', 'null', 'unspecified',
                                    'not provided in the text', 'not found in the text', 
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
                                    'not in the ocr text', 
                                    'Not provided in ocr text',
                                    'not provided in ocr text',
                                    'n/a n/a','n/a, n/a',
                                    'n/a, n/a, n/a','n/a n/a, n/a','n/a, n/a n/a','n/a n/a n/a',
                                    'n/a, n/a, n/a, n/a','n/a n/a n/a n/a','n/a n/a, n/a, n/a','n/a, n/a n/a, n/a','n/a, n/a, n/a n/a',
                                    'n/a n/a n/a, n/a','n/a, n/a n/a n/a',
                                    'n/a n/a, n/a n/a',]:
                    data[key] = ''
            elif isinstance(value, list):
                # Join the list elements into a single string
                data[key] = ', '.join(str(item) for item in value)

        ### align the keys with the template
        # Create a new dictionary with the same order of keys as JSON_dict_structure
        ordered_data = {}

        # This will catch cases where the LLM JSON case does not match the required JSON key's case
        for key in JSON_dict_structure:
            truth_key_lower = key.lower()
            
            llm_value = str(data.get(key, ''))
            if not llm_value:
                llm_value = str(data.get(truth_key_lower, ''))
            
            # Copy the value from data if it exists, else use an empty string
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
