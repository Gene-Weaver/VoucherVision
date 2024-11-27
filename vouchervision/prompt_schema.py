import yaml, json, torch
import outlines
from outlines.generate import json as generate_json
from outlines import models, generate, samplers


class PromptSchema:
    def __init__(self, rules_config_path, model_id):
        self.rules_config_path = rules_config_path
        self.rules_config = self._load_rules_config()
        self.schema = self._build_schema()
        self.prompt_instructions = self._build_prompt_instructions()
        self.model_id = model_id


        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"Using device: {device}")

        self.model = outlines.models.transformers(self.model_id, device=device)
        self.generator = outlines.generate.json(self.model, self.schema)

    def _load_rules_config(self):
        """Load the YAML configuration file."""
        with open(self.rules_config_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return None

    def _build_schema(self):
        """Build a full JSON schema as a string based on the rules section of the config."""
        rules = self.rules_config.get("rules", {})
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


    def _build_prompt_instructions(self):
        """Assemble prompt instructions from the YAML configuration."""
        instructions = self.rules_config.get("instructions", "")
        formatting_instructions = self.rules_config.get("json_formatting_instructions", "")

        return (
            f"Please help me complete this text parsing task given the following rules.\n\n"
            f"The rules are:\n{instructions}\n\n"
            f"{formatting_instructions}\n\n"
            f"Refactor the unstructured OCR text into a JSON dictionary that matches the schema below:\n"
            f"{self.schema}"
        )
    
    def generate_JSON(self, prompt, max_tokens=1024, seed=156733):
        # generator = generate.text(self.model, self.sampler)
        # structured_output = generator(prompt,
        #                    max_tokens=max_tokens, 
        #                     seed=seed,)

        structured_output = self.generator(prompt,
                                        max_tokens=max_tokens, 
                                        seed=seed,
                                        )
        return structured_output


    


# Usage example
if __name__ == "__main__":

    # Path to your YAML configuration
    yaml_path = "D:/Dropbox/VoucherVision/custom_prompts/SLTPvM_default.yaml"
    # OCR text input
    OCR_text = """
    "RANCHO SANTA ANA BOTANIC GARDEN PLANTS OF MEXICO Disticalis specata ( L. ) Greene Nayarit : Isabel Island , E of Tres Marias Is . Volcanic islet -- cormorant and tern breeding ground . In open , level to slightly sloping areas favored by the terns for nesting , near the beach ; surrounded by low forest consisting almost entirely of Crataeva Tapia ( 3-4 m tall ) , Porous soil . CHRISTOPHER DAVIDSON 2060 29 April 1973 Jonvea pilosa ( Presl ) Scribn NIVERSITY THE OF MICH University of Michigan Herbarium MICH 1122841 1817 copyright reserved cm DEC 18 1975 1 122841 "
    """

    # Initialize PromptOutlines
    Prompt_Schema = PromptSchema(yaml_path, model_id="unsloth/mistral-7b-bnb-4bit")
    
    prompt = (f"{Prompt_Schema.prompt_instructions}\n\n"
            f"The unstructured OCR text is:\n{OCR_text}\n\n"
            f"Please produce the JSON dictionary below. Only parse text that appears in the OCR text:"
            )
    structured_output = Prompt_Schema.generate_JSON(prompt)
    print(json.dumps(structured_output, indent=4))

