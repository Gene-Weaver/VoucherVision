from dataclasses import dataclass
import yaml, json


#  catalog = PromptCatalog(OCR="Sample OCR text", domain_knowledge_example="Sample domain knowledge", similarity="0.9")

@dataclass
class PromptCatalog:
    domain_knowledge_example: str = ""
    similarity: str = ""
    OCR: str = ""
    n_fields: int = 0

    # def PROMPT_UMICH_skeleton_all_asia(self, OCR=None, domain_knowledge_example=None, similarity=None):
    def prompt_v1_verbose(self, OCR=None, domain_knowledge_example=None, similarity=None):
        self.OCR = OCR or self.OCR
        self.domain_knowledge_example = domain_knowledge_example or self.domain_knowledge_example
        self.similarity = similarity or self.similarity
        self.n_fields = 22 or self.n_fields

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
        {self.OCR}
        The reference dictionary, which provides an example of the output structure and has an embedding distance of {self.similarity} to the OCR, is:
        {self.domain_knowledge_example}
        Some dictionary fields have special requirements. These requirements specify the format for each field, and are given below:
        {umich_all_asia_rules}
        Please refactor the OCR text into a dictionary, following the rules and the reference structure:
        {structure}
        """

        xlsx_headers = ["Catalog Number","Genus","Species","subspecies","variety","forma","Country","State","County","Locality Name","Min Elevation","Max Elevation","Elevation Units","Verbatim Coordinates","Datum","Cultivated","Habitat","Collectors","Collector Number","Verbatim Date","Date","End Date"]


        return prompt, self.n_fields, xlsx_headers
    
    def prompt_v1_verbose_noDomainKnowledge(self, OCR=None):
        self.OCR = OCR or self.OCR
        self.n_fields = 22 or self.n_fields

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
        {self.OCR}
        Some dictionary fields have special requirements. These requirements specify the format for each field, and are given below:
        {umich_all_asia_rules}
        Please refactor the OCR text into a dictionary, following the rules and the reference structure:
        {structure}
        """

        xlsx_headers = ["Catalog Number","Genus","Species","subspecies","variety","forma","Country","State","County","Locality Name","Min Elevation","Max Elevation","Elevation Units","Verbatim Coordinates","Datum","Cultivated","Habitat","Collectors","Collector Number","Verbatim Date","Date","End Date"]

        return prompt, self.n_fields, xlsx_headers
    
    def prompt_v2_json_rules(self, OCR=None):
        self.OCR = OCR or self.OCR
        self.n_fields = 26 or self.n_fields

        set_rules = """
        1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below.
        2. You should map the unstructured OCR text to the appropriate JSON key and then populate the field based on its rules.
        3. Some JSON key fields are permitted to remain empty if the corresponding information is not found in the unstructured OCR text.
        4. Ignore any information in the OCR text that doesn't fit into the defined JSON structure.
        5. Duplicate dictionary fields are not allowed.
        6. Ensure that all JSON keys are in lowercase.
        7. Ensure that new JSON field values follow sentence case capitalization.
        7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
        8. Ensure the output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
        9. Only return a JSON dictionary represented as a string. You should not explain your answer.
        """

        dictionary_field_format_descriptions = """
        The next section of instructions outlines how to format the JSON dictionary. The keys are the same as those of the final formatted JSON object.
        For each key there is a format requirement that specifies how to transcribe the information for that key. 
        The possible formatting options are:
        1. "verbatim transcription" - field is populated with verbatim text from the unformatted OCR.
        2. "spell check transcription" - field is populated with spelling corrected text from the unformatted OCR.
        3. "boolean yes no" - field is populated with only yes or no.
        4. "integer" - field is populated with only an integer.
        5. "[list]" - field is populated from one of the values in the list.
        6. "yyyy-mm-dd" - field is populated with a date in the format year-month-day.
        The desired null value is also given. Populate the field with the null value of the information for that key is not present in the unformatted OCR text. 
        """

        json_template_rules = """
            {"Dictionary":{
                "catalog_number": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "The barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits."
                },
                "genus": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word 'indet'."
                },
                "species": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to species, do not capitalize species."
                },
                "subspecies": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to subspecies (subsp.)."
                },
                "variety": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to variety (var)."
                },
                "forma": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to form (f.)."
                },
                "country": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Country that corresponds to the current geographic location of collection. Capitalize first letter of each word. If abbreviation is given populate field with the full spelling of the country's name."
                },
                "state": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Administrative division 1 that corresponds to the current geographic location of collection. Capitalize first letter of each word. Administrative division 1 is equivalent to a U.S. State."
                },
                "county": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Administrative division 2 that corresponds to the current geographic location of collection; capitalize first letter of each word. Administrative division 2 is equivalent to a U.S. county, parish, borough."
                },
                "locality_name": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Description of geographic location, landscape, landmarks, regional features, nearby places, or any contextual information aiding in pinpointing the exact origin or site of the specimen."
                },
                "min_elevation": {
                    "format": "integer",
                    "null_value": "",
                    "description": "Minimum elevation or altitude in meters. Only if units are explicit then convert from feet ('ft' or 'ft.' or 'feet') to meters ('m' or 'm.' or 'meters'). Round to integer."
                },
                "max_elevation": {
                    "format": "integer",
                    "null_value": "",
                    "description": "Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are explicit then convert from feet ('ft' or 'ft.' or 'feet') to meters ('m' or 'm.' or 'meters'). Round to integer."
                },
                "elevation_units": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Elevation units must be meters. If min_elevation field is populated, then elevation_units: 'm'. Otherwise elevation_units: ''."
                },
                "verbatim_coordinates": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types are one of [Lat, Long, UTM, TRS]."
                },
                "decimal_coordinates": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format."
                },
                "datum": {
                    "format": "[WGS84, WGS72, WGS66, WGS60, NAD83, NAD27, OSGB36, ETRS89, ED50, GDA94, JGD2011, Tokyo97, KGD2002, TWD67, TWD97, BJS54, XAS80, GCJ-02, BD-09, PZ-90.11, GTRF, CGCS2000, ITRF88, ITRF89, ITRF90, ITRF91, ITRF92, ITRF93, ITRF94, ITRF96, ITRF97, ITRF2000, ITRF2005, ITRF2008, ITRF2014, Hong Kong Principal Datum, SAD69]",
                    "null_value": "",
                    "description": "Datum of location coordinates. Possible values are include in the format list. Leave field blank if unclear."
                },
                "cultivated": {
                    "format": "boolean yes no",
                    "null_value": "",
                    "description": "Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden locations, ornamental, cultivar names, garden, or farm to indicate cultivated plant."
                },
                "habitat": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Description of a plant's habitat or the location where the specimen was collected. Ignore descriptions of the plant itself."
                },
                "plant_description": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Description of plant features such as leaf shape, size, color, stem texture, height, flower structure, scent, fruit or seed characteristics, root system type, overall growth habit and form, any notable aroma or secretions, presence of hairs or bristles, and any other distinguishing morphological or physiological characteristics."
                },
                "collectors": {
                    "format": "verbatim transcription",
                    "null_value": "not present",
                    "description": "Full name(s) of the individual(s) responsible for collecting the specimen. When multiple collectors are involved, their names should be separated by commas."
                },
                "collector_number": {
                    "format": "verbatim transcription",
                    "null_value": "s.n.",
                    "description": "Unique identifier or number that denotes the specific collecting event and associated with the collector."
                },
                "determined_by": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Full name of the individual responsible for determining the taxanomic name of the specimen. Sometimes the name will be near to the characters 'det' to denote determination. This name may be isolated from other names in the unformatted OCR text."
                },
                "multiple_names": {
                    "format": "boolean yes no",
                    "null_value": "",
                    "description": "Indicate whether multiple people or collector names are present in the unformatted OCR text. If you see more than one person's name the value is 'yes'; otherwise the value is 'no'."
                },
                "verbatim_date": {
                    "format": "verbatim transcription",
                    "null_value": "s.d.",
                    "description": "Date of collection exactly as it appears on the label. Do not change the format or correct typos."
                },
                "date": {
                    "format": "yyyy-mm-dd",
                    "null_value": "",
                    "description": "Date the specimen was collected formatted as year-month-day. If specific components of the date are unknown, they should be replaced with zeros. Examples: '0000-00-00' if the entire date is unknown, 'YYYY-00-00' if only the year is known, and 'YYYY-MM-00' if year and month are known but day is not."
                },
                "end_date": {
                    "format": "yyyy-mm-dd",
                    "null_value": "",
                    "description": "If a date range is provided, this represents the later or ending date of the collection period, formatted as year-month-day. If specific components of the date are unknown, they should be replaced with zeros. Examples: '0000-00-00' if the entire end date is unknown, 'YYYY-00-00' if only the year of the end date is known, and 'YYYY-MM-00' if year and month of the end date are known but the day is not."
                },
            },
            "SpeciesName": {
                "taxonomy": [Genus_species]}
            }"""
        
        structure = """{"Dictionary":
                            {
                            "catalog_number": "",
                            "genus": "",
                            "species": "",
                            "subspecies": "",
                            "variety": "",
                            "forma": "",
                            "country": "",
                            "state": "",
                            "county": "",
                            "locality_name": "",
                            "min_elevation": "",
                            "max_elevation": "",
                            "elevation_units": "",
                            "verbatim_coordinates": "",
                            "decimal_coordinates": "",
                            "datum": "",
                            "cultivated": "",
                            "habitat": "",
                            "plant_description": "",
                            "collectors": "",
                            "collector_number": "",
                            "determined_by": "",
                            "multiple_names": "",
                            "verbatim_date":"" ,
                            "date": "",
                            "end_date": ""
                            },
                        "SpeciesName": {"taxonomy": ""}}"""

        prompt = f"""Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
        The rules are:
        {set_rules}
        The unstructured OCR text is:
        {self.OCR}
        {dictionary_field_format_descriptions}
        This is the JSON template that includes instructions for each key:
        {json_template_rules}
        Please populate the following JSON dictionary based on the rules and the unformatted OCR text:
        {structure}
        """

        xlsx_headers = ["catalog_number","genus","species","subspecies","variety","forma","country","state","county","locality_name","min_elevation","max_elevation","elevation_units","verbatim_coordinates","decimal_coordinates","datum","cultivated","habitat","plant_description","collectors","collector_number","determined_by","multiple_names","verbatim_date","date","end_date"]
        
        return prompt, self.n_fields, xlsx_headers
    
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    # These are for dynamically creating your own prompts with n-columns


    def prompt_v2_custom(self, rules_config_path, OCR=None, is_palm=False):
        self.OCR = OCR

        self.rules_config_path = rules_config_path
        self.rules_config = self.load_rules_config()

        self.instructions = self.rules_config['instructions']
        self.json_formatting_instructions = self.rules_config['json_formatting_instructions']

        self.rules_list = self.rules_config['rules']
        self.n_fields = len(self.rules_list['Dictionary'])

        # Set the rules for processing OCR into JSON format
        self.rules = self.create_rules(is_palm)

        self.structure = self.create_structure(is_palm)

        if is_palm:
            prompt = f"""Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
                The rules are:
                {self.instructions}
                The unstructured OCR text is:
                {self.OCR}
                {self.json_formatting_instructions}
                This is the JSON template that includes instructions for each key:
                {self.rules}
                Please populate the following JSON dictionary based on the rules and the unformatted OCR text:
                {self.structure}
                {self.structure}
                {self.structure}
                """
        else:
            prompt = f"""Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
                The rules are:
                {self.instructions}
                The unstructured OCR text is:
                {self.OCR}
                {self.json_formatting_instructions}
                This is the JSON template that includes instructions for each key:
                {self.rules}
                Please populate the following JSON dictionary based on the rules and the unformatted OCR text:
                {self.structure}
                """
        xlsx_headers = self.generate_xlsx_headers(is_palm)
        
        return prompt, self.n_fields, xlsx_headers

    def load_rules_config(self):
        with open(self.rules_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return None

    def create_rules(self, is_palm=False):
        if is_palm:
            # Start with a structure for the "Dictionary" section where each key contains its rules
            dictionary_structure = {
                key: {
                    'description': value['description'],
                    'format': value['format'],
                    'null_value': value.get('null_value', '')
                } for key, value in self.rules_list['Dictionary'].items()
            }

            # Convert the structure to a JSON string without indentation
            structure_json_str = json.dumps(dictionary_structure, sort_keys=False)
            return structure_json_str
        
        else:
            # Start with a structure for the "Dictionary" section where each key contains its rules
            dictionary_structure = {
                key: {
                    'description': value['description'],
                    'format': value['format'],
                    'null_value': value.get('null_value', '')
                } for key, value in self.rules_list['Dictionary'].items()
            }

            # Combine both sections into the overall structure
            full_structure = {
                "Dictionary": dictionary_structure,
                "SpeciesName": self.rules_list['SpeciesName']
            }

            # Convert the structure to a JSON string without indentation
            structure_json_str = json.dumps(full_structure, sort_keys=False)
            return structure_json_str
    
    def create_structure(self, is_palm=False):
        if is_palm:
            # Start with an empty structure for the "Dictionary" section
            dictionary_structure = {key: "" for key in self.rules_list['Dictionary'].keys()}

            # Convert the structure to a JSON string with indentation for readability
            structure_json_str = json.dumps(dictionary_structure, sort_keys=False)
            return structure_json_str
        else:
            # Start with an empty structure for the "Dictionary" section
            dictionary_structure = {key: "" for key in self.rules_list['Dictionary'].keys()}
            
            # Manually define the "SpeciesName" section
            species_name_structure = {"taxonomy": ""}

            # Combine both sections into the overall structure
            full_structure = {
                "Dictionary": dictionary_structure,
                "SpeciesName": species_name_structure
            }

            # Convert the structure to a JSON string with indentation for readability
            structure_json_str = json.dumps(full_structure, sort_keys=False)
            return structure_json_str

    def generate_xlsx_headers(self, is_palm):
        # Extract headers from the 'Dictionary' keys in the JSON template rules
        if is_palm:
            xlsx_headers = list(self.rules_list.keys())
            return xlsx_headers     
        else:
            xlsx_headers = list(self.rules_list["Dictionary"].keys())
            return xlsx_headers
    
    def prompt_v2_custom_redo(self, incorrect_json, is_palm):
        # Load the existing rules and structure
        self.rules_config = self.load_rules_config()
        # self.rules = self.create_rules(is_palm)
        self.structure = self.create_structure(is_palm)
        
        # Generate the prompt using the loaded rules and structure
        if is_palm:
            prompt = f"""The incorrectly formatted JSON dictionary below is not valid. It contains an error that prevents it from loading with the Python command json.loads().
                The incorrectly formatted JSON dictionary below is the literal string returned by a previous function and the error may be caused by markdown formatting.
                You need to return coorect JSON for the following dictionary. Most likely, a quotation mark inside of a field value has not been escaped properly with a backslash.
                Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.
                Escape all JSON control characters that appear in input including ampersand (&) and other control characters. 
                Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
                Ensure the output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
                The incorrectly formatted JSON dictionary: {incorrect_json}
                The output JSON structure: {self.structure}
                The output JSON structure: {self.structure}
                The output JSON structure: {self.structure}
                Please reformat the incorrectly formatted JSON dictionary given the output JSON structure: """
        else:
            prompt = f"""The incorrectly formatted JSON dictionary below is not valid. It contains an error that prevents it from loading with the Python command json.loads().
                The incorrectly formatted JSON dictionary below is the literal string returned by a previous function and the error may be caused by markdown formatting.
                You need to return coorect JSON for the following dictionary. Most likely, a quotation mark inside of a field value has not been escaped properly with a backslash.
                Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.
                Escape all JSON control characters that appear in input including ampersand (&) and other control characters. 
                Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
                Ensure the output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
                The incorrectly formatted JSON dictionary: {incorrect_json}
                The output JSON structure: {self.structure}
                Please reformat the incorrectly formatted JSON dictionary given the output JSON structure: """
        return prompt

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    def prompt_gpt_redo_v1(self, incorrect_json):       
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

        prompt = f"""This text is supposed to be JSON, but it contains an error that prevents it from loading with the Python command json.loads().
                You need to return coorect JSON for the following dictionary. Most likely, a quotation mark inside of a field value has not been escaped properly with a backslash.
                Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.
                Escape all JSON control characters that appear in input including ampersand (&) and other control characters. 
                Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
                Ensure the output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
                The incorrectly formatted JSON dictionary: {incorrect_json}
                The output JSON structure: {structure}
                The refactored JSON disctionary: """
        return prompt
    
    def prompt_gpt_redo_v2(self, incorrect_json):
        structure = """
            {"Dictionary":{
                "catalog_number": "",
                "genus": "",
                "species": "".
                "subspecies": "",
                "variety": "",
                "forma":"",
                "country": "",
                "state": "",
                "county": "",
                "locality_name": "",
                "min_elevation": "",
                "max_elevation": "",
                "elevation_units": "',
                "verbatim_coordinates": "",
                "decimal_coordinates": "",
                "datum": "",
                "cultivated": "",
                "habitat": "",
                "plant_description": "",
                "collectors": "",
                "collector_number": "",
                "determined_by": "",
                "multiple_names": "',
                "verbatim_date": "",
                "date": "",
                "end_date": "",
                },
            "SpeciesName": {"taxonomy": [Genus_species]}}"""
        
        prompt = f"""This text is supposed to be JSON, but it contains an error that prevents it from loading with the Python command json.loads().
                You need to return coorect JSON for the following dictionary. Most likely, a quotation mark inside of a field value has not been escaped properly with a backslash.
                Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.
                Escape all JSON control characters that appear in input including ampersand (&) and other control characters. 
                Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
                Ensure the output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
                The incorrectly formatted JSON dictionary: {incorrect_json}
                The output JSON structure: {structure}
                The refactored JSON disctionary: """
        return prompt
    #####################################################################################################################################
    #####################################################################################################################################
    def prompt_v1_palm2(self, in_list, out_list, OCR=None):
        self.OCR = OCR or self.OCR
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
        "Catalog Number" - {"format": "[barcode]", "null_value": "", "description": the barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits}
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

                input: {self.OCR}

                output:"""
        
        return prompt
    
    def prompt_v1_palm2_noDomainKnowledge(self, OCR=None):
        self.OCR = OCR or self.OCR
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
        "Catalog Number" - {"format": "barcode", "null_value": "", "description": the barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits}
        "Genus" - {"format": "Genus" or "Family indet" if no genus", "null_value": "", "description": taxonomic determination to genus, do captalize genus}
        "Species"- {"format": "species" or "indet" if no species, "null_value": "", "description": taxonomic determination to species, do not captalize species}
        "subspecies" - {"format": "subspecies", "null_value": "", "description": taxonomic determination to subspecies (subsp.)}
        "variety" - {"format": "variety", "null_value": "", "description": taxonomic determination to variety (var)}
        "forma" - {"format": "form", "null_value": "", "description": taxonomic determination to form (f.)}

        "Country" - {"format": "Country", "null_value": "no data", "description": Country that corresponds to the current geographic location of collection; capitalize first letter of each word; use the entire location name even if an abreviation is given}
        "State" - {"format": "Adm. Division 1", "null_value": "no data", "description": Administrative division 1 that corresponds to the current geographic location of collection; capitalize first letter of each word}
        "County" - {"format": "Adm. Division 2", "null_value": "no data", "description": Administrative division 2 that corresponds to the current geographic location of collection; capitalize first letter of each word}
        "Locality Name" - {"format": "verbatim", if no geographic info: "no data provided on label of catalog no: ######", or if illegible: "locality present but illegible/not translated for catalog no: #######", or if no named locality: "no named locality for catalog no: #######", "description": "Description of geographic location or landscape"}

        "Min Elevation" - {format: "elevation integer", "null_value": "","description": Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, default field for elevation if a range is not given}
        "Max Elevation" - {format: "elevation integer", "null_value": "","description": Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, maximum elevation if there are two elevations listed but '' otherwise}
        "Elevation Units" - {format: "m", "null_value": "","description": "m" only if an elevation is present}
        
        "Verbatim Coordinates" - {"format": "Lat, Long, UTM, TRS", "null_value": "", "description": Convert coordinates to Decimal Degrees (D.D°) format, do not use Minutes, Seconds or quotation marks}

        "Datum" - {"format": "WGS84, NAD23 etc.", "null_value": "not present", "description": Datum of coordinates on label; "" is GPS coordinates are not in OCR}
        "Cultivated" - {"format": "yes", "null_value": "", "description": Indicates if specimen was grown in cultivation}
        "Habitat" - {"format": "verbatim", "null_value": "", "description": Description of habitat or location where specimen was collected, ignore descriptions of the plant itself}
        "Collectors" - {"format": "Collector", "null_value": "not present", "description": Full name of person (i.e., agent) who collected the specimen; if more than one person then separate the names with commas}
        "Collector Number" - {"format": "Collector No.", "null_value": "s.n.", "description": Sequential number assigned to collection, associated with the collector}
        "Verbatim Date" - {"format": "verbatim", "null_value": "s.d.", "description": Date of collection exactly as it appears on the label}
        "Date" - {"format": "yyyy-mm-dd", "null_value": "", "description": Date of collection formatted as year, month, and day; zeros may be used for unknown values i.e. 0000-00-00 if no date, YYYY-00-00 if only year, YYYY-MM-00 if no day}
        "End Date" - {"format": "yyyy-mm-dd", "null_value": "", "description": If date range is listed, later date of collection range}
        """
        structure = """{
            "Catalog Number": "",
            "Genus": "",
            "Species": "",
            "subspecies": "",
            "variety": "",
            "forma": "",
            "Country": "",
            "State": "",
            "County": "",
            "Locality Name": "",
            "Min Elevation": "",
            "Max Elevation": "",
            "Elevation Units": "",
            "Verbatim Coordinates": "",
            "Datum": "",
            "Cultivated": "",
            "Habitat": "",
            "Collectors": "",
            "Collector Number": "",
            "Verbatim Date": "",
            "Date": "",
            "End Date": "",
            }"""
        # structure = """{
        #             "Catalog Number": [Catalog Number],
        #             "Genus": [Genus],
        #             "Species": [species],
        #             "subspecies": [subspecies],
        #             "variety": [variety],
        #             "forma": [forma],
        #             "Country": [Country],
        #             "State": [State],
        #             "County": [County],
        #             "Locality Name": [Locality Name],
        #             "Min Elevation": [Min Elevation],
        #             "Max Elevation": [Max Elevation],
        #             "Elevation Units": [Elevation Units],
        #             "Verbatim Coordinates": [Verbatim Coordinates],
        #             "Datum": [Datum],
        #             "Cultivated": [Cultivated],
        #             "Habitat": [Habitat],
        #             "Collectors": [Collectors],
        #             "Collector Number": [Collector Number],
        #             "Verbatim Date": [Verbatim Date],
        #             "Date": [Date],
        #             "End Date": [End Date]
        #             }"""

        prompt = f"""Given the following set of rules:
                set_rules = {set_rules}
                Some dict fields have special requirements listed below. First is the column header. After the - is the format. Do not include the instructions with your response:
                requirements = {umich_all_asia_rules}
                Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.
                The input unformatted OCR text: {self.OCR}
                The output JSON structure: {structure}
                The output JSON structure: {structure}
                The output JSON structure: {structure}
                The refactored JSON disctionary:"""
        
        return prompt
    
    def prompt_v2_palm2(self, OCR=None):
        self.OCR = OCR or self.OCR
        self.n_fields = 26 or self.n_fields

        set_rules = """
        1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below.
        2. You should map the unstructured OCR text to the appropriate JSON key and then populate the field based on its rules.
        3. Some JSON key fields are permitted to remain empty if the corresponding information is not found in the unstructured OCR text.
        4. Ignore any information in the OCR text that doesn't fit into the defined JSON structure.
        5. Duplicate dictionary fields are not allowed.
        6. Ensure that all JSON keys are in lowercase.
        7. Ensure that new JSON field values follow sentence case capitalization.
        8. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
        9. Ensure the output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
        10. Only return a JSON dictionary represented as a string. You should not explain your answer.
        """

        dictionary_field_format_descriptions = """
        The next section of instructions outlines how to format the JSON dictionary. The keys are the same as those of the final formatted JSON object.
        For each key there is a format requirement that specifies how to transcribe the information for that key. 
        The possible formatting options are:
        1. "verbatim transcription" - field is populated with verbatim text from the unformatted OCR.
        2. "spell check transcription" - field is populated with spelling corrected text from the unformatted OCR.
        3. "boolean yes no" - field is populated with only yes or no.
        4. "integer" - field is populated with only an integer.
        5. "[list]" - field is populated from one of the values in the list.
        6. "yyyy-mm-dd" - field is populated with a date in the format year-month-day.
        The desired null value is also given. Populate the field with the null value of the information for that key is not present in the unformatted OCR text. 
        """

        json_template_rules = """
            {
                "catalog_number": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "The barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits."
                },
                "genus": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word 'indet'."
                },
                "species": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to species, do not capitalize species."
                },
                "subspecies": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to subspecies (subsp.)."
                },
                "variety": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to variety (var)."
                },
                "forma": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Taxonomic determination to form (f.)."
                },
                "country": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Country that corresponds to the current geographic location of collection. Capitalize first letter of each word. If abbreviation is given populate field with the full spelling of the country's name. Use sentence-case to capitalize proper nouns."
                },
                "state": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Administrative division 1 that corresponds to the current geographic location of collection. Capitalize first letter of each word. Administrative division 1 is equivalent to a U.S. State. Use sentence-case to capitalize proper nouns."
                },
                "county": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Administrative division 2 that corresponds to the current geographic location of collection; capitalize first letter of each word. Administrative division 2 is equivalent to a U.S. county, parish, borough. Use sentence-case to capitalize proper nouns."
                },
                "locality_name": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Description of geographic location, landscape, landmarks, regional features, nearby places, or any contextual information aiding in pinpointing the exact origin or site of the specimen. Use sentence-case to capitalize proper nouns."
                },
                "min_elevation": {
                    "format": "integer",
                    "null_value": "",
                    "description": "Minimum elevation or altitude in meters. Only if units are explicit then convert from feet ('ft' or 'ft.' or 'feet') to meters ('m' or 'm.' or 'meters'). Round to integer."
                },
                "max_elevation": {
                    "format": "integer",
                    "null_value": "",
                    "description": "Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are explicit then convert from feet ('ft' or 'ft.' or 'feet') to meters ('m' or 'm.' or 'meters'). Round to integer."
                },
                "elevation_units": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Elevation units must be meters. If min_elevation field is populated, then elevation_units: 'm'. Otherwise elevation_units: ''"
                },
                "verbatim_coordinates": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types are one of [Lat, Long, UTM, TRS]."
                },
                "decimal_coordinates": {
                    "format": "spell check transcription",
                    "null_value": "",
                    "description": "Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format."
                },
                "datum": {
                    "format": "[WGS84, WGS72, WGS66, WGS60, NAD83, NAD27, OSGB36, ETRS89, ED50, GDA94, JGD2011, Tokyo97, KGD2002, TWD67, TWD97, BJS54, XAS80, GCJ-02, BD-09, PZ-90.11, GTRF, CGCS2000, ITRF88, ITRF89, ITRF90, ITRF91, ITRF92, ITRF93, ITRF94, ITRF96, ITRF97, ITRF2000, ITRF2005, ITRF2008, ITRF2014, Hong Kong Principal Datum, SAD69]",
                    "null_value": "",
                    "description": "Datum of location coordinates. Possible values are include in the format list. Leave field blank if unclear."
                },
                "cultivated": {
                    "format": "boolean yes no",
                    "null_value": "",
                    "description": "Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden locations, ornamental, cultivar names, garden, or farm to indicate cultivated plant."
                },
                "habitat": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Description of a plant's habitat or the location where the specimen was collected. Ignore descriptions of the plant itself. Use sentence-case to capitalize proper nouns."
                },
                "plant_description": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Description of plant features such as leaf shape, size, color, stem texture, height, flower structure, scent, fruit or seed characteristics, root system type, overall growth habit and form, any notable aroma or secretions, presence of hairs or bristles, and any other distinguishing morphological or physiological characteristics. Use sentence-case to capitalize proper nouns."
                },
                "collectors": {
                    "format": "verbatim transcription",
                    "null_value": "not present",
                    "description": "Full name(s) of the individual(s) responsible for collecting the specimen. Use sentence-case to capitalize proper nouns. When multiple collectors are involved, their names should be separated by commas."
                },
                "collector_number": {
                    "format": "verbatim transcription",
                    "null_value": "s.n.",
                    "description": "Unique identifier or number that denotes the specific collecting event and associated with the collector."
                },
                "determined_by": {
                    "format": "verbatim transcription",
                    "null_value": "",
                    "description": "Full name of the individual responsible for determining the taxanomic name of the specimen. Use sentence-case to capitalize proper nouns. Sometimes the name will be near to the characters 'det' to denote determination. This name may be isolated from other names in the unformatted OCR text."
                },
                "multiple_names": {
                    "format": "boolean yes no",
                    "null_value": "",
                    "description": "Indicate whether multiple people or collector names are present in the unformatted OCR text. Use sentence-case to capitalize proper nouns. If you see more than one person's name the value is 'yes'; otherwise the value is 'no'."
                },
                "verbatim_date": {
                    "format": "verbatim transcription",
                    "null_value": "s.d.",
                    "description": "Date of collection exactly as it appears on the label. Do not change the format or correct typos."
                },
                "date": {
                    "format": "yyyy-mm-dd",
                    "null_value": "",
                    "description": "Date the specimen was collected formatted as year-month-day. If specific components of the date are unknown, they should be replaced with zeros. Examples: '0000-00-00' if the entire date is unknown, 'YYYY-00-00' if only the year is known, and 'YYYY-MM-00' if year and month are known but day is not."
                },
                "end_date": {
                    "format": "yyyy-mm-dd",
                    "null_value": "",
                    "description": "If a date range is provided, this represents the later or ending date of the collection period, formatted as year-month-day. If specific components of the date are unknown, they should be replaced with zeros. Examples: '0000-00-00' if the entire end date is unknown, 'YYYY-00-00' if only the year of the end date is known, and 'YYYY-MM-00' if year and month of the end date are known but the day is not."
                },
            }"""
   
        structure = """{"catalog_number": "",
                        "genus": "",
                        "species": "".
                        "subspecies": "",
                        "variety": "",
                        "forma":"",
                        "country": "",
                        "state": "",
                        "county": "",
                        "locality_name": "",
                        "min_elevation": "",
                        "max_elevation": "",
                        "elevation_units": "',
                        "verbatim_coordinates": "",
                        "decimal_coordinates": "",
                        "datum": "",
                        "cultivated": "",
                        "habitat": "",
                        "plant_description": "",
                        "collectors": "",
                        "collector_number": "",
                        "determined_by": "",
                        "multiple_names": "',
                        "verbatim_date": "",
                        "date": "",
                        "end_date": "",
                        }"""
        # structure = """{"catalog_number": [Catalog Number],
        #                 "genus": [Genus],
        #                 "species": [species],
        #                 "subspecies": [subspecies],
        #                 "variety": [variety],
        #                 "forma": [forma],
        #                 "country": [Country],
        #                 "state": [State],
        #                 "county": [County],
        #                 "locality_name": [Locality Name],
        #                 "min_elevation": [Min Elevation],
        #                 "max_elevation": [Max Elevation],
        #                 "elevation_units": [Elevation Units],
        #                 "verbatim_coordinates": [Verbatim Coordinates],
        #                 "decimal_coordinates": [Decimal Coordinates],
        #                 "datum": [Datum],
        #                 "cultivated": [boolean yes no],
        #                 "habitat": [Habitat Description],
        #                 "plant_description": [Plant Description],
        #                 "collectors": [Name(s) of Collectors],
        #                 "collector_number": [Collector Number],
        #                 "determined_by": [Name(s) of Taxonomist],
        #                 "multiple_names": [boolean yes no],
        #                 "verbatim_date": [Verbatim Date],
        #                 "date": [yyyy-mm-dd],
        #                 "end_date": [yyyy-mm-dd],
        #                 }"""

        prompt = f"""Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
        The rules are:
        {set_rules}
        The unstructured OCR text is:
        {self.OCR}
        {dictionary_field_format_descriptions}
        This is the JSON template that includes instructions for each key:
        {json_template_rules}
        Please populate the following JSON dictionary based on the rules and the unformatted OCR text. The square brackets denote the locations that you should place the new structured text:
        {structure}
        {structure}
        {structure}
        """
        
        return prompt
    
    def prompt_palm_redo_v1(self, incorrect_json):
        structure = """{
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
                    }"""

        prompt = f"""This text is supposed to be JSON, but it contains an error that prevents it from loading with the Python command json.loads().
                You need to return coorect JSON for the following dictionary. Most likely, a quotation mark inside of a field value has not been escaped properly with a backslash.
                Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.
                Escape all JSON control characters that appear in input including ampersand (&) and other control characters. 
                Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
                Ensure the output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
                The incorrectly formatted JSON dictionary: {incorrect_json}
                The output JSON structure: {structure}
                The output JSON structure: {structure}
                The output JSON structure: {structure}
                The refactored JSON disctionary: """
        return prompt
        
    def prompt_palm_redo_v2(self, incorrect_json):
        structure = """{"catalog_number": "",
                        "genus": "",
                        "species": "".
                        "subspecies": "",
                        "variety": "",
                        "forma":"",
                        "country": "",
                        "state": "",
                        "county": "",
                        "locality_name": "",
                        "min_elevation": "",
                        "max_elevation": "",
                        "elevation_units": "',
                        "verbatim_coordinates": "",
                        "decimal_coordinates": "",
                        "datum": "",
                        "cultivated": "",
                        "habitat": "",
                        "plant_description": "",
                        "collectors": "",
                        "collector_number": "",
                        "determined_by": "",
                        "multiple_names": "',
                        "verbatim_date": "",
                        "date": "",
                        "end_date": "",
                        }"""
        
        prompt = f"""This text is supposed to be JSON, but it contains an error that prevents it from loading with the Python command json.loads().
                You need to return coorect JSON for the following dictionary. Most likely, a quotation mark inside of a field value has not been escaped properly with a backslash.
                Given the input, please generate a JSON response. Please note that the response should not contain any special characters, including quotation marks (single ' or double \"), within the JSON values.
                Escape all JSON control characters that appear in input including ampersand (&) and other control characters. 
                Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
                Ensure the output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
                The incorrectly formatted JSON dictionary: {incorrect_json}
                The output JSON structure: {structure}
                The output JSON structure: {structure}
                The output JSON structure: {structure}
                The refactored JSON disctionary: """
        return prompt