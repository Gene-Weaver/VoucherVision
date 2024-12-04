from pydantic import BaseModel, Field, create_model
from typing import List, Optional
import yaml
import outlines
from outlines.generate import json as generate_json

'''
DON'T USE THIS, USE THE prompt_outlines.py VERSION INSTEAD
'''

class PromptOutlines:
    def __init__(self, rules_config_path):
        self.rules_config_path = rules_config_path
        self.rules_config = self._load_rules_config()
        self.instructions = self.rules_config['instructions']
        self.json_formatting_instructions = self.rules_config['json_formatting_instructions']
        self.rules_list = self.rules_config['rules']
        self.model = None
        self.generator = None
        self._create_pydantic_model()

    def _load_rules_config(self):
        """Load the YAML rules configuration file."""
        with open(self.rules_config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return None

    def _create_pydantic_model(self):
        """Dynamically create a Pydantic model based on the rules."""
        fields = {
            key: (Optional[str], Field(default='', description=value))
            for key, value in self.rules_list.items()
        }

        # Create the dynamic Pydantic model using create_model
        self.PromptModel = create_model('PromptModel', **fields)

    def set_model(self, model_name):
        """Set the language model for Outlines."""
        self.model = outlines.models.transformers(model_name)

    def initialize_generator(self):
        """Initialize the JSON generator with the dynamic Pydantic model."""
        if not self.model:
            raise ValueError("Model not set. Call set_model() before initializing the generator.")
        self.generator = generate_json(self.model, self.PromptModel)

    def create_prompt(self, OCR_text):
        """Create a complex LLM prompt based on the YAML instructions."""
        cleaned_OCR_text = self._sanitize_text(OCR_text)
        prompt = (
            f"Please help me complete this text parsing task given the following rules and unstructured OCR text. "
            f"Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure "
            f"specified in the following rules. Please follow the rules strictly.\n"
            f"The rules are:\n{self.instructions}\n\n"
            f"{self.json_formatting_instructions}\n\n"
            f"The unstructured OCR text is:\n{cleaned_OCR_text}\n\n"
            f"Please populate the following JSON dictionary based on the rules and the unformatted OCR text."
        )
        return prompt

    def generate_structured_output(self, OCR_text):
        """Generate structured JSON output based on the prompt and schema."""
        if not self.generator:
            raise ValueError("Generator not initialized. Call initialize_generator() before generating output.")
        prompt = self.create_prompt(OCR_text)
        return self.generator(prompt)

    @staticmethod
    def _sanitize_text(text):
        """Sanitize text to remove colons and double quotes."""
        return text.replace(":", "").replace("\"", "")



# Usage example
if __name__ == "__main__":
    # Path to your YAML configuration
    yaml_path = "D:/Dropbox/VoucherVision/custom_prompts/SLTPvM_default.yaml"

    # Initialize PromptOutlines
    prompt_outlines = PromptOutlines(yaml_path)

    # Set the LLM model (replace with your model name)
    prompt_outlines.set_model("mistralai/Mistral-7B-Instruct-v0.3")

    # Initialize the generator
    prompt_outlines.initialize_generator()

    # OCR text input
    OCR_text = """
    catalogNumber 123456
     Quercus rubra
    Quercus
    rubra
    collectedBy: John Doe, Jane Smith
    Ann arbor michigan USA
    """

    # Generate structured output
    structured_output = prompt_outlines.generate_structured_output(OCR_text)
    print(structured_output)
