class OCRPromptCatalog:
    """
    A catalog of OCR prompts for different versions and use cases.
    This class provides a central repository for storing and retrieving prompts.
    """
    def __init__(self):
        # Dictionary to store prompts by version
        self.prompts = {
            "default_plus_minorcorrect_idhandwriting": """Please perform OCR on this scientific image and extract all of the words and text verbatim. Then correct any minor typos for scientific species names. Identify any handwritten text. Do not explain your answer, only return the verbatim text:""",

            "default_plus_minorcorrect": """Please perform OCR on this scientific image and extract all of the words and text verbatim. Then correct any minor typos for scientific species names. Do not explain your answer, only return the verbatim text:""",

            "default": """Please perform OCR on this scientific image and extract all of the words and text verbatim. Do not explain your answer, only return the verbatim text:""",

            "handwriting_only": """Identify and extract all handwritten text from this image. Do not include printed text.Only return the handwritten text verbatim:""",

            "species_only": """Extract all scientific species names from this image. Do not include any other text. Correct minor typos in the species names if needed. Return only the species names:""",

            "detailed_metadata": """Extract all text from this scientific image, including metadata, species names, and handwritten notes. Correct any minor typos in species names and categorize the extracted text by type (metadata, species names, handwritten notes). Return the result in JSON format:""",
        }

    def get_prompt(self, version="default"):
        """
        Retrieve the prompt for a specific version.
        
        :param version: The version key for the desired prompt.
        :return: The corresponding prompt as a string.
        :raises ValueError: If the version does not exist.
        """
        if version not in self.prompts:
            raise ValueError(f"Prompt version '{version}' not found in the catalog.")
        return self.prompts[version]

    def add_prompt(self, version, prompt):
        """
        Add a new prompt to the catalog.
        
        :param version: The version key for the new prompt.
        :param prompt: The prompt text to store.
        :raises ValueError: If the version already exists.
        """
        if version in self.prompts:
            raise ValueError(f"Prompt version '{version}' already exists.")
        self.prompts[version] = prompt

    def list_versions(self):
        """
        List all available prompt versions.
        
        :return: A list of version keys.
        """
        return list(self.prompts.keys())
    
    def get_prompts_by_keys(self, keys):
        """
        Retrieve a list of prompts for the given list of keys.
        
        :param keys: A list of version keys for the desired prompts.
        :return: A list of corresponding prompts.
        :raises ValueError: If any key in the list does not exist.
        """
        missing_keys = [key for key in keys if key not in self.prompts]
        if missing_keys:
            raise ValueError(f"Prompt versions not found in the catalog: {', '.join(missing_keys)}")
        return [self.prompts[key] for key in keys]
