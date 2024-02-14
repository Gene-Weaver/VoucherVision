from dataclasses import dataclass
from langchain_core.pydantic_v1 import Field, create_model
import yaml, json, os, shutil

@dataclass
class PromptCatalog:
    domain_knowledge_example: str = ""
    similarity: str = ""
    OCR: str = ""
    n_fields: int = 0

    
    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################
    # These are for dynamically creating your own prompts with n-columns


    def prompt_SLTP(self, rules_config_path, OCR=None, is_palm=False):
        self.OCR = OCR

        self.rules_config_path = rules_config_path
        self.rules_config = self.load_rules_config()

        self.instructions = self.rules_config['instructions']
        self.json_formatting_instructions = self.rules_config['json_formatting_instructions']

        self.rules_list = self.rules_config['rules']
        self.n_fields = len(self.rules_config['rules'])

        # Set the rules for processing OCR into JSON format
        self.rules = self.create_rules(is_palm)

        self.structure, self.dictionary_structure = self.create_structure(is_palm)

        ''' between  instructions and json_formatting_instructions. Made the prompt too long. Better performance without it
        The unstructured OCR text is:
        {self.OCR}
        '''
        if is_palm:
            prompt = f"""Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
                The rules are:
                {self.instructions}
                {self.json_formatting_instructions}
                This is the JSON template that includes instructions for each key:
                {self.rules}
                The unstructured OCR text is:
                {self.OCR}
                Please populate the following JSON dictionary based on the rules and the unformatted OCR text:
                {self.structure}
                {self.structure}
                {self.structure}
                """
        else:
            prompt = f"""Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
                The rules are:
                {self.instructions}
                {self.json_formatting_instructions}
                This is the JSON template that includes instructions for each key:
                {self.rules}
                The unstructured OCR text is:
                {self.OCR}
                Please populate the following JSON dictionary based on the rules and the unformatted OCR text:
                {self.structure}
                """
        # xlsx_headers = self.generate_xlsx_headers(is_palm)
        
        # return prompt, self.PromptJSONModel, self.n_fields, xlsx_headers
        return prompt, self.dictionary_structure


    def copy_prompt_template_to_new_dir(self, new_directory_path, rules_config_path):
        # Ensure the target directory exists, create it if it doesn't
        if not os.path.exists(new_directory_path):
            os.makedirs(new_directory_path)
        
        # Define the path for the new file location
        new_file_path = os.path.join(new_directory_path, os.path.basename(rules_config_path))
        
        # Copy the file to the new location
        try:
            shutil.copy(rules_config_path, new_file_path)
            print(f"Prompt [{os.path.basename(rules_config_path)}] copied successfully to {new_file_path}")
        except Exception as exc:
            print(f"Error copying [{os.path.basename(rules_config_path)}] file: {exc}")


    def load_rules_config(self):
        with open(self.rules_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return None

    def create_rules(self, is_palm=False):
        dictionary_structure = {key: value for key, value in self.rules_list.items()}

        # Convert the structure to a JSON string without indentation
        structure_json_str = json.dumps(dictionary_structure, sort_keys=False)
        return structure_json_str
    
    def create_structure(self, is_palm=False):
        # Create fields for the Pydantic model dynamically
        fields = {key: (str, Field(default=value, description=value)) for key, value in self.rules_list.items()}

        # Dynamically create the Pydantic model
        DynamicJSONParsingModel = create_model('SLTPvA', **fields)
        DynamicJSONParsingModel_use = DynamicJSONParsingModel()

        # Define the structure for the "Dictionary" section
        dictionary_fields = {key: (str, Field(default='', description="")) for key in self.rules_list.keys()}
        
        # Dynamically create the "Dictionary" Pydantic model
        PromptJSONModel = create_model('PromptJSONModel', **dictionary_fields)

        # Convert the model to JSON string (for demonstration)
        dictionary_structure = PromptJSONModel().dict()
        structure_json_str = json.dumps(dictionary_structure, sort_keys=False, indent=4)
        return structure_json_str, dictionary_structure


    def generate_xlsx_headers(self, is_palm):
        # Extract headers from the 'Dictionary' keys in the JSON template rules
        if is_palm:
            xlsx_headers = list(self.rules_list.keys())
            return xlsx_headers     
        else:
            xlsx_headers = list(self.rules_list.keys())
            return xlsx_headers
    
