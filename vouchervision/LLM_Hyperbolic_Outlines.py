from pydantic import BaseModel, create_model, ConfigDict
import json, time, random, os
import outlines
try:
    from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, save_individual_prompt, sanitize_prompt
    from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template
except:
    from utils_LLM import SystemLoadMonitor, run_tools, save_individual_prompt, sanitize_prompt
    from utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template
    
class HyperbolicHandler:
    RETRY_DELAY = 2  # Wait 2 seconds before retrying
    MAX_RETRIES = 5  # Maximum number of retries
    STARTING_TEMP = 0.5
    RANDOM_SEED = 2023

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation):
        self.cfg = cfg
        if self.cfg is None:
            self.tool_WFO = False
            self.tool_GEO = False
            self.tool_wikipedia = False
        else: 
            self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
            self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
            self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']

            self.logger = logger
            self.monitor = SystemLoadMonitor(logger)

        self.JSON_dict_structure = JSON_dict_structure

        self.api_key = os.getenv("HYPERBOLIC_API_KEY")
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}

        self.config_vals_for_permutation = config_vals_for_permutation

        # Initialize the Outlines model
        # self.model = models.openai(self.model_name, api_key=self.api_key)
        self.model = outlines.models.openai(
            self.model_name,
            base_url= "https://api.hyperbolic.xyz/v1/",
            api_key=self.api_key
            )
        self._set_config()

        # Dynamically create the Pydantic model
        # self.OutputSchema = self._create_dynamic_pydantic_model(self.JSON_dict_structure)
        self.schema = self._build_schema()

        # self.generator = outlines.generate.text(self.model)
        self.generator = outlines.generate.json(self.model, self.schema)

    def generate_JSON(self, prompt, max_tokens=1024, seed=156733):
        structured_output = self.generator(prompt,
                                        max_tokens=max_tokens, 
                                        )
        return structured_output
    
    def _build_schema(self):
        """Build a full JSON schema as a string based on the rules section of the config."""
        rules = self.JSON_dict_structure
        properties = {
            key: {"type": "string", "description": value} for key, value in rules.items()
        }

        schema = {
            "title": "Generated Schema",
            "type": "object",
            "properties": properties,
            "required": list(rules.keys()),  # Include all keys as required
        }

        # Serialize the schema to a JSON string
        return json.dumps(schema)

    def _set_config(self):
        """Set configuration values dynamically."""
        if self.config_vals_for_permutation:
            self.starting_temp = self.config_vals_for_permutation.get('hyperbolic', {}).get('temperature', self.STARTING_TEMP)
            self.config = {
                'max_tokens': self.config_vals_for_permutation.get('hyperbolic', {}).get('max_tokens', 1024),
                'temperature': self.starting_temp,
                'top_p': self.config_vals_for_permutation.get('hyperbolic', {}).get('top_p', 0.9),
                'random_seed': self.config_vals_for_permutation.get('hyperbolic', {}).get('random_seed', self.RANDOM_SEED),
            }
        else:
            self.starting_temp = self.STARTING_TEMP
            self.config = {
                'max_tokens': 1024,
                'temperature': self.starting_temp,
                'top_p': 0.9,
                'random_seed': self.RANDOM_SEED,
            }
        self.temp_increment = 0.2
        self.adjust_temp = self.starting_temp

    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        self.config['random_seed'] = random.randint(1, 1000)
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp} and random_seed to {self.config["random_seed"]}')
        self.adjust_temp += self.temp_increment
        self.config['temperature'] = self.adjust_temp

    def _reset_config(self):
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp} and random_seed to {self.RANDOM_SEED}')
        self.adjust_temp = self.starting_temp
        self.config['temperature'] = self.starting_temp
        self.config['random_seed'] = self.RANDOM_SEED

    @staticmethod
    def _create_dynamic_pydantic_model(json_structure):
        """Create a dynamic Pydantic model based on the keys in the JSON_dict_structure."""
        fields = {key: (str, '') for key in json_structure.keys()}  # All fields are strings with default ''
        return create_model("OutputSchema", **fields, model_config=ConfigDict(extra="forbid"))

    def call_llm_api_Hyperbolic(self, prompt_template, json_report, paths):
        _, _, _, _, _, json_file_path_wiki, txt_file_path_ind_prompt = paths

        self.json_report = json_report
        if self.json_report:
            self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        self.monitor.start_monitoring_usage()

        for attempt in range(self.MAX_RETRIES):
            try:
                # Generate structured JSON output
                
                # generator = JsonGenerator(self.model, self.OutputSchema, **self.config)
                output = self.generate_JSON(prompt_template)
                # print(json.dumps(output, indent=4))
                # output = self.generator(prompt_template)

                # Validate and align JSON
                output_dict = validate_and_align_JSON_keys_with_template(output, self.JSON_dict_structure)
                if output_dict is None:
                    self.logger.error(f'[Attempt {attempt + 1}] Failed to extract valid JSON')
                    self._adjust_config()
                    continue

                nt_in = self.generator.prompt_tokens
                nt_out = self.generator.completion_tokens

                self.monitor.stop_inference_timer()  # Starts tool timer too
                if self.json_report:
                    self.json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                output_WFO, WFO_record, output_GEO, GEO_record = run_tools(output_dict, self.tool_WFO, self.tool_GEO, self.tool_wikipedia, json_file_path_wiki)

                save_individual_prompt(sanitize_prompt(prompt_template), txt_file_path_ind_prompt)

                self.logger.info(f"Formatted JSON Pre-Sanitize:\n{json.dumps(output_dict, indent=4)}")

                usage_report = self.monitor.stop_monitoring_report_usage()

                if self.adjust_temp != self.starting_temp:
                    self._reset_config()

                if self.json_report:
                    self.json_report.set_text(text_main=f'LLM call successful')
                return output_dict, nt_in, nt_out, WFO_record, GEO_record, usage_report

            except Exception as e:
                self.logger.error(f'Error during JSON generation: {e}')
                self._adjust_config()
                time.sleep(self.RETRY_DELAY)

        self.logger.info(f"Failed to extract valid JSON after [{self.MAX_RETRIES}] attempts")
        if self.json_report:
            self.json_report.set_text(text_main=f'Failed to extract valid JSON after [{self.MAX_RETRIES}] attempts')

        self.monitor.stop_inference_timer()
        usage_report = self.monitor.stop_monitoring_report_usage()
        self._reset_config()
        if self.json_report:
            self.json_report.set_text(text_main=f'LLM call failed')

        return None, 0, 0, None, None, usage_report


if __name__ == '__main__':
    prompt_test = """Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
                The rules are:
                1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below. 2. Map the unstructured OCR text to the appropriate JSON key and populate the field given the user-defined rules. 3. JSON key values are permitted to remain empty strings if the corresponding information is not found in the unstructured OCR text. 4. 
Duplicate dictionary fields are not allowed. 5. Ensure all JSON keys are in camel case. 6. Ensure new JSON field values follow sentence case capitalization. 7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template. 8. Ensure output JSON string is valid JSON format. It should not have trailing commas or unquoted keys. 9. Only return a JSON dictionary represented as a string. You should not explain your answer. 10. I expect the output to be only the complete JSON object without explanation.
                This section provides rules for formatting each JSON value organized by the JSON key.
                This is the JSON template that includes instructions for each key:
                {"catalogNumber": "Barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits.", "order": "The full scientific name of the order in which 
the taxon is classified. Order must be capitalized.", "family": "The full scientific name of the family in which the taxon is classified. Family must be capitalized.", "scientificName": "The scientific name of the taxon including genus, specific epithet, and any lower classifications.", "scientificNameAuthorship": "The authorship information for the scientificName formatted according to the conventions of the applicable Darwin Core nomenclaturalCode.", "genus": "Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word 'indet'.", "subgenus": "The full scientific name of the subgenus in which the taxon is classified. Values should include the genus to avoid homonym confusion.", "specificEpithet": "The name of the first or species epithet of the scientificName. Only include the species epithet.", "infraspecificEpithet": "The name of the lowest or terminal infraspecific epithet of the scientificName, excluding any rank designation.", "identifiedBy": "A comma separated list of names of people, groups, or organizations who assigned the taxon to the subject organism. This is not the specimen collector.", "recordedBy": "A comma separated list of names of people, groups, or organizations responsible for observing, recording, collecting, or presenting the original specimen. The primary collector or observer should be listed first.", "recordNumber": "An identifier given to 
the occurrence at the time it was recorded. Often serves as a link between field notes and an occurrence record, such as a specimen collector's number.", "verbatimEventDate": "The verbatim original representation of the date and time information for when the specimen was collected. Date of collection exactly as it appears on the label. Do not change the format or 
correct typos.", "eventDate": "Date the specimen was collected formatted as year-month-day, YYYY-MM_DD. If specific components of the date are unknown, they should be replaced with zeros. Examples \"0000-00-00\" if the entire date is unknown, \"YYYY-00-00\" if only the year is known, and \"YYYY-MM-00\" if year and month are known but day is not.", "habitat": "A category or description of the habitat in which the specimen collection event occurred.", "occurrenceRemarks": "Text describing the specimen's geographic location. Text describing the appearance of the specimen. A statement about the presence or absence of a taxon at a the collection location. Text describing the significance of the specimen, such as a specific expedition or notable collection. Description of plant features such as leaf shape, size, color, stem texture, height, flower structure, scent, fruit or seed characteristics, root system 
type, overall growth habit and form, any notable aroma or secretions, presence of hairs or bristles, and any other distinguishing morphological or physiological characteristics.", "country": "The name of the country or major administrative unit in which the specimen was originally collected.", "stateProvince": "The name of the next smaller administrative region than country (state, province, canton, department, region, etc.) in which the specimen was originally collected.", "county": "The full, unabbreviated name of the next smaller administrative region than stateProvince (county, shire, department, parish etc.) in which the specimen was originally collected.", "municipality": "The full, unabbreviated name of the next smaller administrative region than county (city, municipality, etc.) in which the specimen was originally collected.", "locality": "Description of geographic location, landscape, landmarks, regional features, nearby places, or any contextual information aiding in pinpointing the exact origin or location of the specimen.", "degreeOfEstablishment": "Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden locations, ornamental, cultivar names, garden, or farm to indicate cultivated plant. Use either - unknown or cultivated.", "decimalLatitude": "Latitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format.", "decimalLongitude": "Longitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format.", "verbatimCoordinates": "Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types include [Lat, Long, UTM, TRS].", "minimumElevationInMeters": "Minimum elevation or altitude in meters. Only if units are explicit then convert from feet (\"ft\" or \"ft.\"\" or \"feet\") to meters (\"m\" or \"m.\" or \"meters\"). Round to integer.", "maximumElevationInMeters": "Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are 
explicit then convert from feet (\"ft\" or \"ft.\" or \"feet\") to meters (\"m\" or \"m.\" or \"meters\"). Round to integer."}
                The unstructured OCR text is:
                OCR

Google Handwritten OCR
RANCHO SANTA ANA BOTANIC GARDEN
DistichtiLANTEN MET (L.) Greene
OF MEXICO
Nayarit Isabel Island, E of Tres Marias Is.
Volcanic islet--cormorant and tern breeding
ground.
THE
In open, level to slightly sloping areas
favored by the terns for nesting, hear the
beach; surrounded by low forest consisting
almost entirely of Crataeva Tapia (3-4 m tall),
Porous soil.
MICH
UNIVERSITY OF MICHIC
1817
copyright reserved 122841 cm
29 April 1973
DEC 1 8 1975
University of Michigan Herbarium
1122841
                Please populate the following JSON dictionary based on the rules and the unformatted OCR text. I expect the output to be only the complete JSON object without explanation:
                {'catalogNumber': '', 'order': '', 'family': '', 'scientificName': '', 'scientificNameAuthorship': '', 'genus': '', 'subgenus': '', 'specificEpithet': '', 'infraspecificEpithet': '', 'identifiedBy': '', 'recordedBy': '', 'recordNumber': '', 'verbatimEventDate': '', 'eventDate': '', 'habitat': '', 'occurrenceRemarks': '', 'country': '', 'stateProvince': '', 'county': '', 'municipality': '', 'locality': '', 'degreeOfEstablishment': '', 'decimalLatitude': '', 'decimalLongitude': '', 'verbatimCoordinates': '', 'minimumElevationInMeters': '', 'maximumElevationInMeters': ''}"""
    JSON_dict_structure = {"catalogNumber": "", "order": "", "family": "", "scientificName": "", "scientificNameAuthorship": "", "genus": "", "subgenus": "", "specificEpithet": "", "infraspecificEpithet": "", "identifiedBy": "", "recordedBy": "", "recordNumber": "", "verbatimEventDate": "", "eventDate": "", "habitat": "", "occurrenceRemarks": "", "country": "", "stateProvince": "", "county": "", "municipality": "", "locality": "", "degreeOfEstablishment": "", "decimalLatitude": "", "decimalLongitude": "", "verbatimCoordinates": "", "minimumElevationInMeters": "", "maximumElevationInMeters": ""}
    
    os.environ["HYPERBOLIC_API_KEY"] = ""
    
    models = [
                'mistralai/Pixtral-12B-2409',
                'Qwen/Qwen2-VL-7B-Instruct',
                'Qwen/Qwen2-VL-72B-Instruct', # PASS
                'Qwen/Qwen2.5-72B-Instruct', # PASS
                'Qwen/QwQ-32B-Preview',
                'Qwen/Qwen2.5-Coder-32B-Instruct', # PASS
                'meta-llama/Llama-3.2-3B-Instruct', # PASS
                'meta-llama/Meta-Llama-3.1-405B-Instruct',
                'meta-llama/Meta-Llama-3.1-405B-FP8',
                'meta-llama/Meta-Llama-3.1-8B-Instruct', # PASS
                'meta-llama/Meta-Llama-3.1-70B-Instruct', # PASS
                'meta-llama/Meta-Llama-3-70B-Instruct',
                'deepseek-ai/DeepSeek-V2.5',
        ]
    for model in models:
        try:
            ml = HyperbolicHandler(None, None, model, JSON_dict_structure, None)
            # ml = HyperbolicHandler(None, None, "meta-llama/Meta-Llama-3-70B-Instruct", JSON_dict_structure, None)
            # prompt_test = prompt_test.replace('\'', '\"')
            output = ml.generate_JSON(prompt_test)
            print(f"SUCCESS >>> {model}")
            print(json.dumps(output, indent=4))
        except Exception as e:
            print(f"            FAILED >>> {model}")
            print(f"            {e}")


