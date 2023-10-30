'''
###################################################################
##########################  chatGPT  ##############################
###################################################################

Prompts for chatGPT
    PROMPT_UMICH_skeleton_all_asia
        - Designed for the "All Asia" TCN at the University of Michigan Herbarium
        - Has 21 columns for a skeleton record

    PROMPT_OCR_Organized
        - Designed to privide human transcribers text that is grouped by category
          so that QC of automated transcription is faster. This output is sent to
          a custom QC GUI where human labelers can simply copy and paste raw text
          (but organized by category) into fields that may have been transcribed
          incorrectly by the LLM. 

###################################################################
##########################  chatGPT  ##############################
###################################################################
'''

def PROMPT_UMICH_skeleton_all_asia(OCR, domain_knowledge_example, similarity):
    set_rules = """1. Your job is to return a new dict based on the structure of the reference dict ref_dict and these are your rules. 
                    2. You must look at ref_dict and refactor the new text called OCR to match the same formatting. 
                    3. OCR contains unstructured text inside of [], use your knowledge to put the OCR text into the correct ref_dict column. 
                    4. If OCR is mostly empty and contains substantially less text than the ref_dict examples, then only return "None" and skip all other steps.
                    5. If there is a field that does not have a direct proxy in the OCR text, you can fill it in based on your knowledge, but you cannot generate new information.
                    6. Never put text from the ref_dict values into the new dict, but you must use the headers from ref_dict. 
                    7. There cannot be duplicate dictionary fields.
                    8. Only return the new dict, do not explain your answer."""

    umich_all_asia_rules = """
    "Catalog Number" - {"format": "[Catalog Number]", "null_value": "", "description": The barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits}
    "Genus" - {"format": "[Genus]" or "[Family] indet" if no genus", "null_value": "", "description": taxonomic determination to genus, do captalize genus}
    "Species"- {"format": "[species]" or "indet" if no species, "null_value": "", "description": taxonomic determination to species, do not captalize species}
    "subspecies" - {"format": "[subspecies]", "null_value": "", "description": taxonomic determination to subspecies (subsp.)}
    "variety" - {"format": "[variety]", "null_value": "", "description": taxonomic determination to variety (var)}
    "forma" - {"format": "[form]", "null_value": "", "description": taxonomic determination to form (f.)}

    "Country" - {"format": "[Country]", "null_value": "no data", "description": Country that corresponds to the current geographic location of collection; capitalize first letter of each word; use the entire location name even if an abreviation is given}
    "State" - {"format": "[Adm. Division 1]", "null_value": "no data", "description": Administrative division 1 that corresponds to the current geographic location of collection; capitalize first letter of each word}
    "County" - {"format": "[Adm. Division 2]", "null_value": "no data", "description": Administrative division 2 that corresponds to the current geographic location of collection; capitalize first letter of each word}
    "Locality Name" - {"format": "verbatim", if no geographic info: "no data provided on label of catalog no: [######]", or if illegible: "locality present but illegible/not translated for catalog no: #######", or if no named locality: "no named locality for catalog no: #######", "description": "Description of geographic location or landscape"}

    "Min Elevation" - {format: "elevation integer", "null_value": "","description": Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, default field for elevation if a range is not given}
    "Max Elevation" - {format: "elevation integer", "null_value": "","description": Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, maximum elevation if there are two elevations listed but '' otherwise}
    "Elevation Units" - {format: "m", "null_value": "","description": "m" only if an elevation is present}
    
    "Verbatim Coordinates" - {"format": "[Lat, Long | UTM | TRS]", "null_value": "", "description": Verbatim coordinates as they appear on the label, fix typos to match standardized GPS coordinate format}

    "Datum" - {"format": "[WGS84, NAD23 etc.]", "null_value": "not present", "description": Datum of coordinates on label; "" is GPS coordinates are not in OCR}
    "Cultivated" - {"format": "yes", "null_value": "", "description": Indicates if specimen was grown in cultivation}
    "Habitat" - {"format": "verbatim", "null_value": "", "description": Description of habitat or location where specimen was collected, ignore descriptions of the plant itself}
    "Collectors" - {"format": "[Collector]", "null_value": "not present", "description": Full name of person (i.e., agent) who collected the specimen; if more than one person then separate the names with commas}
    "Collector Number" - {"format": "[Collector No.]", "null_value": "s.n.", "description": Sequential number assigned to collection, associated with the collector}
    "Verbatim Date" - {"format": "verbatim", "null_value": "s.d.", "description": Date of collection exactly as it appears on the label}
    "Date" - {"format": "[yyyy-mm-dd]", "null_value": "", "description": Date of collection formatted as year, month, and day; zeros may be used for unknown values i.e. 0000-00-00 if no date, YYYY-00-00 if only year, YYYY-MM-00 if no day}
    "End Date" - {"format": "[yyyy-mm-dd]", "null_value": "", "description": If date range is listed, later date of collection range}
    """

    structure = """{"Dictionary":
                        {
                        "Catalog Number": [Catalog Number],
                        "Genus": [Genus],
                        "Species": [Species],
                        "subspecies": [subspecies],
                        "variety": [variety],
                        "forma": [forma],
                        "Country": [Country],
                        "State": [State],
                        "County": [County],
                        "Locality Name": [Locality Name],
                        "Min Elevation": [Min Elevation],
                        "Max Elevation": [Max Elevation],
                        "Elevation Units": [Elevation Units],
                        "Verbatim Coordinates": [Verbatim Coordinates],
                        "Datum": [Datum],
                        "Cultivated": [Cultivated],
                        "Habitat": [Habitat],
                        "Collectors": [Collectors],
                        "Collector Number": [Collector Number],
                        "Verbatim Date": [Verbatim Date],
                        "Date": [Date],
                        "End Date": [End Date]
                        },
                    "SpeciesName": {"taxonomy": "genus_species"}}"""

    prompt = f"""Given the following set of rules:

            set_rules = {set_rules}

            The following is the raw OCR text that you must translate into a properly formatted Python dictionary based on the rules:

            OCR = {OCR}

            The following is an example dictionary that has an embedding distance of {similarity} compared to OCR. Use if as a guide, but never copy text directly from the domain_knowledge:

            domain_knowledge = {domain_knowledge_example}

            Some dict fields have special requirements listed below. First is the column header. After the - is the format. Do not include the instructions with your response:

            requirements = {umich_all_asia_rules}

            Please transform the OCR text into a Python dictionary following the rules to complete this dictionary, replace [] with content:
            formatted_ocr = {structure}"""
    # print(f'{OCR}\n\n')
    # print(f'{domain_knowledge_example}\n\n')
    return prompt


def PROMPT_OCR_Organized(OCR):
    set_rules = """1. Your job is to parse messy text and return a new dict based on these rules. 
                    2. The messy text is similar to the information contained in Darwin Core Archive files for herbarium specimens. 
                    3. You need to bin the text into 4 different information categories including TAXONOMY, GEOGRAPHY, LOCALITY, COLLECTING and MISCELLANEOUS based on symantics.
                    4. Within each information category list separate discrete content with the comma seperator ",".
                    5. Denote discrete content inside of each subcategory with quotation marks, like this "discrete content".
                    6. If you can provide more detailed information for the GEOGRAPHY category, such as a more thorough location hierarchy, please include additional information along with the verbatim transcriptions.
                    7. Transcribe verbatim unless there is a typo. You can correct typos and misspellings and you can adjust capitalization of the letters in content words to fit standard conventions given the context.
                    8. If some content listed in the descriptions below are not present in OCR, the just skip those subfields. 
                    9. Only return the new dict, do not explain your answer."""
    
    structure = """{"Dictionary":{
                        "TAXONOMY": ["taxonomic topics", "more taxonomic information",],
                        "GEOGRAPHY": ["geographic topics","more geographic information",],
                        "LOCALITY": ["location topics", "more location information",], 
                        "COLLECTING": ["documentation and collection topics", "more documentation and collection information",],
                        "MISCELLANEOUS": ["miscellaneous topics", "more remaining miscellaneous info",]
                        },
                "Summary": ["one sentence description of content"]}"""
    category_rules = """
    "TAXONOMY" - Information to include: all content relating to the name of the plant species including Order, Family, Genus, Species, Subspecies, Variety, and Forma.
    "GEOGRAPHY" - Information to include: The government defined names of places that would appear on a map of political boundaries including Countries, States, Prefectures, Provinces, Districts, Counties, Cities, or Adminstrative Divisions. Adjust capitalization to follow standard conventions for each. 
    "LOCALITY" - Information to include: descriptions of the landscape, habitat, surroundings or nearby places including towns, roads, buildings, geologic features, and distances.
    "COLLECTING" - Information to include: the names of the people who collected the specimen; the collector's number; the verbatim date; all dates translated int the format [yyyy-mm-dd] with zeros replacing unknown numbers; anything relating to cultivation status or whether it was grown in a garden or captive setting; all descriptions of the habitat where the plant grows or information about the way the plant looks and behaves.
    "MISCELLANEOUS" - Information to include: any leftover text that does not fit into the previous categories. 
    "Summary" - The second of two required keys in the output dict, fomatted_ocr. A brief one sentence summary of the content.
    """

    prompt = f"""Given the following set of rules:
            set_rules = {set_rules}
            You must parse the OCR content into the following formatted dictionary:
            structure = {structure}
            The following is the raw OCR text that you must reformat into a properly formatted Python dictionary based on the set_rules:
            OCR = {OCR}
            The following are descriptions of what information to bin into each TAXONOMY, GEOGRAPHY, LOCALITY, COLLECTING, and MISCELLANEOUS category, plus the Summary:
            descriptions = {category_rules}
            Please transform the OCR text into a Python dictionary following the rules to complete this dictionary:
            formatted_ocr = """
    # print(f'{OCR}\n\n')
    return prompt

### GPT4 edited PROMPT_UMICH_skeleton_all_asia to creat the following prompt:
def PROMPT_UMICH_skeleton_all_asia_GPT4(OCR, domain_knowledge_example, similarity):
    set_rules = """
    Please note that your task is to generate a dictionary, following the below rules:
    1. Refactor the unstructured OCR text into a dictionary based on the reference dictionary structure (ref_dict).
    2. Each field of OCR corresponds to a column of the ref_dict. You should correctly map the values from OCR to the respective fields in ref_dict.
    3. If the OCR is mostly empty and contains substantially less text than the ref_dict examples, then only return "None".
    4. If there is a field in the ref_dict that does not have a corresponding value in the OCR text, fill it based on your knowledge but don't generate new information.
    5. Do not use any text from the ref_dict values in the new dict, but you must use the headers from ref_dict.
    6. Duplicate dictionary fields are not allowed.
    7. Only return the new dictionary. You should not explain your answer.
    8. Your output should be a Python dictionary represented as a JSON string.
    """

    umich_all_asia_rules = """{
        "Catalog Number": {
            "format": "[Catalog Number]",
            "null_value": "",
            "description": "The barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits"
        },
        "Genus": {
            "format": "[Genus] or '[Family] indet' if no genus",
            "null_value": "",
            "description": "Taxonomic determination to genus, do capitalize genus"
        },
        "Species": {
            "format": "[species] or 'indet' if no species",
            "null_value": "",
            "description": "Taxonomic determination to species, do not capitalize species"
        },
        "subspecies": {
            "format": "[subspecies]",
            "null_value": "",
            "description": "Taxonomic determination to subspecies (subsp.)"
        },
        "variety": {
            "format": "[variety]",
            "null_value": "",
            "description": "Taxonomic determination to variety (var)"
        },
        "forma": {
            "format": "[form]",
            "null_value": "",
            "description": "Taxonomic determination to form (f.)"
        },
        "Country": {
            "format": "[Country]",
            "null_value": "",
            "description": "Country that corresponds to the current geographic location of collection; capitalize first letter of each word; use the entire location name even if an abbreviation is given"
        },
        "State": {
            "format": "[Adm. Division 1]",
            "null_value": "",
            "description": "Administrative division 1 that corresponds to the current geographic location of collection; capitalize first letter of each word"
        },
        "County": {
            "format": "[Adm. Division 2]",
            "null_value": "",
            "description": "Administrative division 2 that corresponds to the current geographic location of collection; capitalize first letter of each word"
        },
        "Locality Name": {
            "format": "verbatim, if no geographic info: 'no data provided on label of catalog no: [######]', or if illegible: 'locality present but illegible/not translated for catalog no: #######', or if no named locality: 'no named locality for catalog no: #######'",
            "description": "Description of geographic location or landscape"
        },
        "Min Elevation": {
            "format": "elevation integer",
            "null_value": "",
            "description": "Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, default field for elevation if a range is not given"
        },
        "Max Elevation": {
            "format": "elevation integer",
            "null_value": "",
            "description": "Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, maximum elevation if there are two elevations listed but '' otherwise"
        },
        "Elevation Units": {
            "format": "m",
            "null_value": "",
            "description": "'m' only if an elevation is present"
        },
        "Verbatim Coordinates": {
            "format": "[Lat, Long | UTM | TRS]",
            "null_value": "",
            "description": "Verbatim coordinates as they appear on the label, fix typos to match standardized GPS coordinate format"
        },
        "Datum": {
            "format": "[WGS84, NAD23 etc.]",
            "null_value": "",
            "description": "GPS Datum of coordinates on label; empty string "" if GPS coordinates are not in OCR"
        },
        "Cultivated": {
            "format": "yes",
            "null_value": "",
            "description": "Indicates if specimen was grown in cultivation"
        },
        "Habitat": {
            "format": "verbatim",
            "null_value": "",
            "description": "Description of habitat or location where specimen was collected, ignore descriptions of the plant itself"
        },
        "Collectors": {
            "format": "[Collector]",
            "null_value": "not present",
            "description": "Full name of person (i.e., agent) who collected the specimen; if more than one person then separate the names with commas"
        },
        "Collector Number": {
            "format": "[Collector No.]",
            "null_value": "s.n.",
            "description": "Sequential number assigned to collection, associated with the collector"
        },
        "Verbatim Date": {
            "format": "verbatim",
            "null_value": "s.d.",
            "description": "Date of collection exactly as it appears on the label"
        },
        "Date": {
            "format": "[yyyy-mm-dd]",
            "null_value": "",
            "description": "Date of collection formatted as year, month, and day; zeros may be used for unknown values i.e., 0000-00-00 if no date, YYYY-00-00 if only year, YYYY-MM-00 if no day"
        },
        "End Date": {
            "format": "[yyyy-mm-dd]",
            "null_value": "",
            "description": "If date range is listed, later date of collection range"
        }
    }"""

    structure = """{"Dictionary":
                        {
                        "Catalog Number": [Catalog Number],
                        "Genus": [Genus],
                        "Species": [species],
                        "subspecies": [subspecies],
                        "variety": [variety],
                        "forma": [forma],
                        "Country": [Country],
                        "State": [State],
                        "County": [County],
                        "Locality Name": [Locality Name],
                        "Min Elevation": [Min Elevation],
                        "Max Elevation": [Max Elevation],
                        "Elevation Units": [Elevation Units],
                        "Verbatim Coordinates": [Verbatim Coordinates],
                        "Datum": [Datum],
                        "Cultivated": [Cultivated],
                        "Habitat": [Habitat],
                        "Collectors": [Collectors],
                        "Collector Number": [Collector Number],
                        "Verbatim Date": [Verbatim Date],
                        "Date": [Date],
                        "End Date": [End Date]
                        },
                    "SpeciesName": {"taxonomy": [Genus_species]}}"""

    prompt = f"""I'm providing you with a set of rules, an unstructured OCR text, and a reference dictionary (domain knowledge). Your task is to convert the OCR text into a structured dictionary that matches the structure of the reference dictionary. Please follow the rules strictly.
    The rules are as follows:
    {set_rules}
    The unstructured OCR text is:
    {OCR}
    The reference dictionary, which provides an example of the output structure and has an embedding distance of {similarity} to the OCR, is:
    {domain_knowledge_example}
    Some dictionary fields have special requirements. These requirements specify the format for each field, and are given below:
    {umich_all_asia_rules}
    Please refactor the OCR text into a dictionary, following the rules and the reference structure:
    {structure}
    """

    return prompt


### GPT4 edited PROMPT_OCR_Organized prompt:
def PROMPT_OCR_Organized_GPT4(OCR):
    set_rules = """
    You need to parse a messy text and return a new dictionary, based on the following rules:
    1. The messy text is similar to the information contained in Darwin Core Archive files for herbarium specimens. 
    2. You need to organize the text into 4 different information categories: TAXONOMY, GEOGRAPHY, LOCALITY, COLLECTING, and MISCELLANEOUS. Use semantic analysis to do so.
    3. Separate discrete content within each category with a comma separator "," and denote it with quotation marks, like this "discrete content".
    4. When the content falls under the GEOGRAPHY category and more detailed information is available, include the additional information.
    5. Transcribe the OCR text verbatim unless there is a typo. Correct any typos or misspellings and adjust the capitalization of the letters in content words to fit standard conventions.
    6. The output should follow the structure given in 'structure'. If the content described in the descriptions below isn't present in the OCR text, just skip those subfields.
    7. Your output should only be the new dictionary. You should not explain your answer.
    8. The output should include a 'Summary' section, providing a brief one-sentence overview of the OCR text content. This should be a general summary, touching upon the main points from all categories.
    """
    
    category_rules = """{
        "TAXONOMY": {
            "description": "Include all content that pertains to the name of the plant species, such as Order, Family, Genus, Species, Subspecies, Variety, and Forma."
        },
        "GEOGRAPHY": {
            "description": "Include names of places that are government-defined and appear on a map with political boundaries, such as Countries, States, Prefectures, Provinces, Districts, Counties, Cities, or Administrative Divisions. Adjust capitalization to follow standard conventions."
        },
        "LOCALITY": {
            "description": "Include descriptions of the immediate surroundings or physical landscape, including features such as roads, buildings, landmarks, natural formations, and proximities to towns. Avoid including geopolitical names that fall under the 'GEOGRAPHY' category."
        },
        "COLLECTING": {
            "description": "Include names of the people who collected the specimen; the collector's number; the verbatim date; any dates translated into the format [yyyy-mm-dd] with zeros replacing unknown numbers; details relating to cultivation status or if it was grown in a garden or captive setting; all descriptions of the habitat where the plant grows or information about the plant's appearance and behavior."
        },
        "MISCELLANEOUS": {
            "description": "Include any additional text that does not fit into the previous categories and does not relate directly to any other specified categories."
        },
        "Summary": {
            "description": "The second of two required keys in the output dictionary, 'formatted_ocr'. This should provide a concise, one-sentence summary of the content."
        }
    }"""
    
    structure = """
    {
        "Dictionary": {
            "TAXONOMY": {
                "Order": "",
                "Family": "",
                "Genus": "",
                "Species": "",
                "Subspecies": "",
                "Variety": "",
                "Forma": ""
            },
            "GEOGRAPHY": {
                "Country": "",
                "State": "",
                "Prefecture": "",
                "Province": "",
                "District": "",
                "County": "",
                "City": "",
                "Administrative Division": ""
            },
            "LOCALITY": {
                "Landscape": "",
                "Nearby Places": ""
            },
            "COLLECTING": {
                "Collector": "",
                "Collector's Number": "",
                "Verbatim Date": "",
                "Formatted Date": "",
                "Cultivation Status": "",
                "Habitat Description": ""
            },
            "MISCELLANEOUS": {
                "Additional Information": ""
            }
        },
        "Summary": {
            "Content Summary": ""
        }
    }
    """

    prompt = f"""
    I'm providing you with a set of rules, and an unstructured OCR text. Your task is to convert the OCR text into a structured dictionary, organized by several categories. Please follow the rules strictly.

    The rules are as follows:

    {set_rules}

    The unstructured OCR text that needs to be restructured is:

    {OCR}

    The information should be organized into the following categories:

    {category_rules}

    The structure of the output dictionary should be as follows:

    {structure}

    Please transform the OCR text into a dictionary, following these rules and the provided structure.
    """
    
    return prompt

'''
###################################################################
#########################  PaLM  ##################################
###################################################################

Prompts for PaLM
    PROMPT_PaLM_UMICH_skeleton_all_asia
        - Designed for the "All Asia" TCN at the University of Michigan Herbarium
        - Has 21 columns for a skeleton record

    PROMPT_PaLM_Redo
        - PaLM version (2023/06) routinely puts quotation marks inside
          dictionary fields without escaping the character:
                correct: \"
                incorrect: "
          These appear in GPS coordinates. If json.loads() cannot parse 
          the original output, then PROMPT_PaLM_Redo is triggered, telling
          PaLM to reformat the JSON string. Usually one redo call will suffice.

    PROMPT_PaLM_OCR_Organized
        - Similar to the chatGPT version.
        - Designed to privide human transcribers text that is grouped by category
          so that QC of automated transcription is faster. This output is sent to
          a custom QC GUI where human labelers can simply copy and paste raw text
          (but organized by category) into fields that may have been transcribed
          incorrectly by the LLM. 


###################################################################
#########################  PaLM  ##################################
###################################################################
'''
def PROMPT_PaLM_UMICH_skeleton_all_asia(OCR, in_list, out_list):
    set_rules = """1. Your job is to return a new dict based on the structure of the reference dict ref_dict and these are your rules. 
                    2. You must look at ref_dict and refactor the new text called OCR to match the same formatting. 
                    3. OCR contains unstructured text inside of [], use your knowledge to put the OCR text into the correct ref_dict column. 
                    4. If OCR is mostly empty and contains substantially less text than the ref_dict examples, then only return "None" and skip all other steps.
                    5. If there is a field that does not have a direct proxy in the OCR text, you can fill it in based on your knowledge, but you cannot generate new information.
                    6. Never put text from the ref_dict values into the new dict, but you must use the headers from ref_dict. 
                    7. There cannot be duplicate dictionary fields.
                    8. Only return the new dict, do not explain your answer.
                    9. Do not include quotation marks in content, only use quotation marks to represent values in dictionaries.
                    10. For GPS coordinates only use Decimal Degrees (D.D°)
                    11. "Given the input text, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values."""

    umich_all_asia_rules = """
    "Genus" - {"format": "[Genus]" or "[Family] indet" if no genus", "null_value": "", "description": taxonomic determination to genus, do captalize genus}
    "Species"- {"format": "[species]" or "indet" if no species, "null_value": "", "description": taxonomic determination to species, do not captalize species}
    "subspecies" - {"format": "[subspecies]", "null_value": "", "description": taxonomic determination to subspecies (subsp.)}
    "variety" - {"format": "[variety]", "null_value": "", "description": taxonomic determination to variety (var)}
    "forma" - {"format": "[form]", "null_value": "", "description": taxonomic determination to form (f.)}

    "Country" - {"format": "[Country]", "null_value": "no data", "description": Country that corresponds to the current geographic location of collection; capitalize first letter of each word; use the entire location name even if an abreviation is given}
    "State" - {"format": "[Adm. Division 1]", "null_value": "no data", "description": Administrative division 1 that corresponds to the current geographic location of collection; capitalize first letter of each word}
    "County" - {"format": "[Adm. Division 2]", "null_value": "no data", "description": Administrative division 2 that corresponds to the current geographic location of collection; capitalize first letter of each word}
    "Locality Name" - {"format": "verbatim", if no geographic info: "no data provided on label of catalog no: [######]", or if illegible: "locality present but illegible/not translated for catalog no: #######", or if no named locality: "no named locality for catalog no: #######", "description": "Description of geographic location or landscape"}

    "Min Elevation" - {format: "elevation integer", "null_value": "","description": Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, default field for elevation if a range is not given}
    "Max Elevation" - {format: "elevation integer", "null_value": "","description": Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, maximum elevation if there are two elevations listed but '' otherwise}
    "Elevation Units" - {format: "m", "null_value": "","description": "m" only if an elevation is present}
    
    "Verbatim Coordinates" - {"format": "[Lat, Long | UTM | TRS]", "null_value": "", "description": Convert coordinates to Decimal Degrees (D.D°) format, do not use Minutes, Seconds or quotation marks}

    "Datum" - {"format": "[WGS84, NAD23 etc.]", "null_value": "not present", "description": Datum of coordinates on label; "" is GPS coordinates are not in OCR}
    "Cultivated" - {"format": "yes", "null_value": "", "description": Indicates if specimen was grown in cultivation}
    "Habitat" - {"format": "verbatim", "null_value": "", "description": Description of habitat or location where specimen was collected, ignore descriptions of the plant itself}
    "Collectors" - {"format": "[Collector]", "null_value": "not present", "description": Full name of person (i.e., agent) who collected the specimen; if more than one person then separate the names with commas}
    "Collector Number" - {"format": "[Collector No.]", "null_value": "s.n.", "description": Sequential number assigned to collection, associated with the collector}
    "Verbatim Date" - {"format": "verbatim", "null_value": "s.d.", "description": Date of collection exactly as it appears on the label}
    "Date" - {"format": "[yyyy-mm-dd]", "null_value": "", "description": Date of collection formatted as year, month, and day; zeros may be used for unknown values i.e. 0000-00-00 if no date, YYYY-00-00 if only year, YYYY-MM-00 if no day}
    "End Date" - {"format": "[yyyy-mm-dd]", "null_value": "", "description": If date range is listed, later date of collection range}
    """

    prompt = f"""Given the following set of rules:

            set_rules = {set_rules}

            Some dict fields have special requirements listed below. First is the column header. After the - is the format. Do not include the instructions with your response:

            requirements = {umich_all_asia_rules}

            Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.

            input: {in_list[0]}

            output: {out_list[0]}

            input: {in_list[1]}

            output: {out_list[1]}

            input: {in_list[2]}

            output: {out_list[2]}

            input: {OCR}

            output:"""
    
    return prompt
            # input: {in_list[3]}

            # output: {out_list[3]}

def PROMPT_PaLM_OCR_Organized(OCR):
    set_rules = """1. Your job is to parse messy text and return a new dict based on these rules. 
                    2. The messy text is similar to the information contained in Darwin Core Archive files for herbarium specimens. 
                    3. You need to bin the text into 4 different information categories including TAXONOMY, GEOGRAPHY, LOCALITY, COLLECTING and MISCELLANEOUS based on symantics.
                    4. Within each information category list separate discrete content with the comma seperator ",".
                    5. Denote discrete content inside of each subcategory with quotation marks, like this "discrete content".
                    6. If you can provide more detailed information for the GEOGRAPHY category, such as a more thorough location hierarchy, please include additional information along with the verbatim transcriptions.
                    7. Transcribe verbatim unless there is a typo. You can correct typos and misspellings and you can adjust capitalization of the letters in content words to fit standard conventions given the context.
                    8. Do not include quotation marks in content, only use quotation marks to represent values in dictionaries.
                    9. For GPS coordinates only use Decimal Degrees (D.D°) do not use Minutes, Seconds, or quotation marks.
                    10. If some content listed in the descriptions below are not present in OCR, the just skip those subfields. 
                    11. Only return the new dict, do not explain your answer."""
    

    structure = """{"TAXONOMY": {"taxonomic topic": "relevant taxonomic info", "another taxonomic topic": "relevant taxonomic info"},
                    "GEOGRAPHY": {"geographic topic": "relevant geographic info","another geographic topic": "more relevant geographic info"},
                    "LOCALITY": {"location topic": "relevant location info","another location topic": "relevant location info"}, 
                    "COLLECTING": {"documentation topic": "relevant documentation info", "another documentation topic": "documentation info"},
                    "MISCELLANEOUS": {"miscellaneous topic": "remaining miscellaneous info", "another miscellaneous topic": "more remaining miscellaneous info"}}"""


    category_rules = """"TAXONOMY" - Information to include: all content relating to the name of the plant species including Order, Family, Genus, Species, Subspecies, Variety, and Forma.

    "GEOGRAPHY" - Information to include: The government defined names of places that would appear on a map of political boundaries including Countries, States, Prefectures, Provinces, Districts, Counties, Cities, or Adminstrative Divisions. Adjust capitalization to follow standard conventions for each. 
    
    "LOCALITY" - Information to include: descriptions of the landscape, habitat, surroundings or nearby places including towns, roads, buildings, geologic features, and distances.
    
    "COLLECTING" - Information to include: the names of the people who collected the specimen; the collector's number; the verbatim date; all dates translated int the format [yyyy-mm-dd] with zeros replacing unknown numbers; anything relating to cultivation status or whether it was grown in a garden or captive setting; all descriptions of the habitat where the plant grows or information about the way the plant looks and behaves.
    
    "MISCELLANEOUS" - Information to include: any leftover text that does not fit into the previous categories."""


    ex_1_out = """{"TAXONOMY": {"species": "Quercus Robur C.","common name": "Ch\u00eane Robur."},"GEOGRAPHY": {"location": "Bord d'un chemin, Rymenam (Ann.), BRUXELLES"},"LOCALITY": {"distance": "300 centimeters","surroundings": "Botanic Garden, Golden Thread","miscellaneous": "I, 500 200, inches, 600 300, 700 400, is per inch (opticn), 800 500, 850 550"},"COLLECTING": {"date": "1867-07-07","collection number": "Acc. 1919","habitat": "HERBIER DU JARDIN BOTANIQUE DE L'\u00c9TAT"},"MISCELLANEOUS": {}}"""
    ex_1_in = """Quercus Robur C. Ch\u00eane Robur.  Bord d'un chemin, Rymenam (Ann.), BRUXELLES  300 centimeters   surroundings  Botanic Garden, Golden Thread I, 500 200, inches, 600 300, 700 400, is per inch (opticn), 800 500, 850 550  1867 July 7  HERBIER DU JARDIN BOTANIQUE DE L'\u00c9TAT 1919"""

    ex_2_out = """{"TAXONOMY": {"species": "Brookea tomentosa Benth."},"GEOGRAPHY": {"country": "Malaysia","stateProvince": "Sabah","county": "Beaufort District","verbatimLocality": "Beaufort Hill. 5\u00b022'N, 115\u00b045'E. Elev. 200 m.","higherGeography": "Crocker Formation"},"LOCALITY": {"habitat": "Burned logged dipterocarp forest."},"COLLECTING": {"recordedBy": "John H. Beaman","recordNumber": "6844","verbatimEventDate": "28 August 1983","eventDate": "1983-08-28","country": "UNITED STATES","cultivationStatus": "wild","associatedTaxa": "Reed S. Beaman and Teofila E. Beaman"},"MISCELLANEOUS": {"additionalData": "Herbaria of Michigan State University (MSC) and Universiti Kebangsaan Malaysia, Sabah Campus (UKMS), centimeter, 3539788, inches, PLANTS OF BORNEO, 500 200, 600 300, US, Institution, Smithsonian, 700 400, 1, 800 500, 850 550, \u0160 8"}}"""
    ex_2_in = """Brookea tomentosa Benth. Malaysia  Sabah  Beaufort District Beaufort Hill. 5\u00b022'N, 115\u00b045'E.  200 m. Crocker Formation habitat Burned logged dipterocarp forest. John H. Beaman 6844 28 August 1983 1983-08-28 UNITED STATES Reed S. Beaman and Teofila E. Beaman Herbaria of Michigan State University (MSC) and Universiti Kebangsaan Malaysia, Sabah Campus (UKMS), centimeter, 3539788, inches, PLANTS OF BORNEO, 500 200, 600 300, US, Institution, Smithsonian, 700 400, 1, 800 500, 850 550, \u0160 8"}}"""


    prompt = f"""Given the following set of rules:

            set_rules = {set_rules}

            You must parse the OCR content into the following formatted dictionary:

            structure = {structure}

            The following are descriptions of what information to bin into each TAXONOMY, GEOGRAPHY, LOCALITY, COLLECTING, and MISCELLANEOUS category, plus the Summary:

            descriptions = {category_rules}

            Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.

            For all field values you must properly escape quotation marks using a backslash so that JSON formatting is maintained.

            Escape all JSON control characters that appear in input including ampersand (&) and other control characters. 

            input: {ex_1_in}

            output: {ex_1_out}

            input: {ex_2_in}

            output: {ex_2_out}

            input: {OCR}

            output:"""
    
    return prompt


def PROMPT_PaLM_Redo(bad_response):
    # GPS coordinates are the problem, so skip them
    ex_1_in = """{"TAXONOMY": {"species": "Quercus Robur C.", "common name": "Ch\u00eane Robur."} "GEOGRAPHY": "location": "Bord d'un chemin, Rymenam (Ann.), BRUXELLES"}, "LOCALITY": {"distance": "300 centimeters", "surroundings": "Botanic Garden, Golden Thread", "miscellaneous": "I, 500 200, inches, 600 300, 700 400, is per inch (opticn), 800 500, 850 550"}, "COLLECTING": {"date": "1867-07-07", "collection number": "Acc. 1919", "habitat": "HERBIER DU JARDIN BOTANIQUE DE L'\u00c9TAT"}, "MISCELLANEOUS": {}}"""
    ex_1_out = """{"TAXONOMY": {"species": "Quercus Robur C.", "common name": "Ch\u00eane Robur."}, "GEOGRAPHY": {"location": "Bord d'un chemin, Rymenam (Ann.), BRUXELLES"}, "LOCALITY": {"distance": "300 centimeters", "surroundings": "Botanic Garden, Golden Thread", "miscellaneous": "I, 500 200, inches, 600 300, 700 400, is per inch (opticn), 800 500, 850 550"}, "COLLECTING": {"date": "1867-07-07", "collection number": "Acc. 1919", "habitat": "HERBIER DU JARDIN BOTANIQUE DE L'\u00c9TAT"}, "MISCELLANEOUS": {}}"""

    '''for just skipping the verbatim coordinates'''
    ex_2_in = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "vari"ety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    ex_2_out = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "variety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    
    ex_3_in = """{"Genus": "Forchammeria" "Species": "Watsonii", "subspecies": "", "variety": "",  "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    ex_3_out = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "variety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    
    ex_4_in = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "variety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "" "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    ex_4_out = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "variety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    
    '''for trying to fix the escape chars'''
    # ex_2_in = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "vari"ety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Verbatim Coordinates": "", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    # ex_2_out = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "variety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Verbatim Coordinates": "", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    
    # ex_3_in = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "variety": "",  "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Verbatim Coordinates": "-34°6'15"N, 119°45'0"W",, "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    # ex_3_out = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "variety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Verbatim Coordinates": "-34°6\'15\"N, 119°45\'0\"W",, "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    
    # ex_4_in = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "variety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Verbatim Coordinates": "-25°3'24"N, 109°50'0"W", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    # ex_4_out = """{"Genus": "Forchammeria", "Species": "Watsonii", "subspecies": "", "variety": "", "forma": "", "Country": "Mexico", "State": "Baja California Sur", "County": "Cerralvo Island", "Locality Name": "South end of Cerralvo Island", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Verbatim Coordinates": "-25°3\'24\"N, 109°50\'0\"W",, "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Reid Moran", "Collector Number": "3592", "Verbatim Date": "3. April... 1952", "End Date": ""}"""
    
    prompt = f"""This text is supposed to be JSON, but it contains an error that prevents it from loading with the Python command json.loads().
                
                You need to return coorect JSON for the following dictionary. Most likely, a quotation mark inside of a field value has not been escaped properly with a backslash.
                
                Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.
                
                Escape all JSON control characters that appear in input including ampersand (&) and other control characters. 
                
                input: {ex_1_in}

                output: {ex_1_out}

                input: {ex_2_in}

                output: {ex_2_out}

                input: {ex_3_in}

                output: {ex_3_out}

                input: {ex_4_in}

                output: {ex_4_out}

                input: {bad_response}

                output:"""

    return prompt


def PROMPT_JSON(opt, bad_response=''):
    if opt == 'dict':

        guide = f"""This is the JSON text that contains an error, typically there is an errant quotation mark inside of value, so escape with a backslash any quotation marks that occur in the middle of a value field:
                {bad_response}"""
        
        structure = """Below is the correct JSON formatting. Modify the text to conform to the following format, fixing the incorrect JSON:
        {"Dictionary":
            {
            "Catalog Number": [Catalog Number],
            "Genus": [Genus],
            "Species": [species],
            "subspecies": [subspecies],
            "variety": [variety],
            "forma": [forma],
            "Country": [Country],
            "State": [State],
            "County": [County],
            "Locality Name": [Locality Name],
            "Min Elevation": [Min Elevation],
            "Max Elevation": [Max Elevation],
            "Elevation Units": [Elevation Units],
            "Verbatim Coordinates": [Verbatim Coordinates],
            "Datum": [Datum],
            "Cultivated": [Cultivated],
            "Habitat": [Habitat],
            "Collectors": [Collectors],
            "Collector Number": [Collector Number],
            "Verbatim Date": [Verbatim Date],
            "Date": [Date],
            "End Date": [End Date]
            },
        "SpeciesName": {"taxonomy": [Genus_species]}}"""
        prompt = '\n'.join([guide, structure])
        return prompt
    
    elif opt == 'helper':
        guide = f"""This is the JSON text that contains an 1error, typically there is an errant quotation mark inside of value, so escape with a backslash any quotation marks that occur in the middle of a value field:
                {bad_response}"""
        
        structure = """Below is the correct JSON formatting. Modify the text to conform to the following format, fixing the incorrect JSON:
            {
                "Dictionary": {
                    "TAXONOMY": {
                        "Order": "",
                        "Family": "",
                        "Genus": "",
                        "Species": "",
                        "Subspecies": "",
                        "Variety": "",
                        "Forma": ""
                    },
                    "GEOGRAPHY": {
                        "Country": "",
                        "State": "",
                        "Prefecture": "",
                        "Province": "",
                        "District": "",
                        "County": "",
                        "City": "",
                        "Administrative Division": ""
                    },
                    "LOCALITY": {
                        "Landscape": "",
                        "Nearby Places": ""
                    },
                    "COLLECTING": {
                        "Collector": "",
                        "Collector's Number": "",
                        "Verbatim Date": "",
                        "Formatted Date": "",
                        "Cultivation Status": "",
                        "Habitat Description": ""
                    },
                    "MISCELLANEOUS": {
                        "Additional Information": ""
                    }
                },
                "Summary": {
                    "Content Summary": ""
                }
            }
            """
    prompt = '\n'.join([guide, structure])
    return prompt




