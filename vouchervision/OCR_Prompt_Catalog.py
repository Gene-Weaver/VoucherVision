class OCRPromptCatalog:
    """
    A catalog of OCR prompts for different versions and use cases.
    This class provides a central repository for storing and retrieving prompts.
    """
    def __init__(self):
        # Dictionary to store prompts by version
        self.prompts = {
            "anthropic": """Please perform OCR on this scientific image and extract all of the words and text verbatim. Do not explain, present, or qualify your answer, only provide the text in your final response because I know the context already. I also need you to annotate the verbatim text following these steps: 1. Place section sign symbols on either side of text that has been stricken, crossed out, or redacted because these are likely to be old species names (§crossed text§) 2. Identify any handwritten text by placing guillemet quote symbols on either side («handwritten text») 3. Correct any minor typos for scientific species names 4. Carefully assess whether there are multiple barcodes in the image that represent multiple specimens on the same herbarium sheet; if there are multiple barcodes in the image, then include "[MULTIPLE BARCODES]" otherwise just skip this step 5. Ignore rulers 6. Maintain the relative text positions by using indentation and paragraph breaks 7. Do not explain your answer, only return the final text with the appropriate annotations. The final text should not be markdown. Finally, do not explain, present, or qualify your answer, only provide the text in your final response. Do not provide a description of the image.""",

            "default_plus_minorcorrect_excludestricken_idhandwriting": """Please perform OCR on this scientific image and extract all of the words and text verbatim, but exclude text that has been stricken, crossed out, or redacted. Also correct any minor typos for scientific species names. Identify any handwritten text. Do not explain your answer, only return the verbatim text:""",
            
            "default_plus_minorcorrect_addressstricken_idhandwriting": """Please perform OCR on this scientific image and extract all of the words and text verbatim, but place three tildes on either side of text that has been stricken, crossed out, or redacted because these are likely to be old species names. Also correct any minor typos for scientific species names. Identify any handwritten text by placing three asterisks symbols on either side. If there are multiple barcodes in the image, then include this in the reported OCR '[MULTIPLE BARCODES]'.Do not explain your answer, only return the verbatim text with the appropriate annotations:""",
            
            "verbatim_with_annotations": """Please perform OCR on this scientific image and extract all of the words and text verbatim. I also need you to annotate the verbatim text following these steps:
1. Place section sign symbols on either side of text that has been stricken, crossed out, or redacted because these are likely to be old species names (§crossed text§)
2. Identify any handwritten text by placing guillemet quote symbols on either side («handwritten text»)
3. Correct any minor typos for scientific species names
4. Carefully assess whether there are multiple barcodes in the image that represent multiple specimens on the same herbarium sheet; if there are multiple barcodes in the image, then include "[MULTIPLE BARCODES]" otherwise just skip this step
5. Ignore rulers
6. Maintain the relative text positions by using indentation and paragraph breaks
7. Do not explain your answer, only return the final text with the appropriate annotations. The final text should not be markdown.""",

            "default_plus_minorcorrect_excludestricken_gpt4": """Please perform OCR on this scientific image and extract all of the words and text verbatim, excluding text that has been stricken, crossed out, or redacted. Use your knowledge and the context of the surrounding text to also correct any minor typos caused by poor OCR for scientific species names. Your corrections should be minimal and should focus character recognition errors. The correction cannot have more or fewer characters than the original word. Do not explain your answer, only return the verbatim text:""",
            
            "default_plus_minorcorrect_idhandwriting": """Please perform OCR on this scientific image and extract all of the words and text verbatim, but also correct any minor typos for scientific species names. Identify any handwritten text. Do not explain your answer, only return the verbatim text:""",
            
            "default_plus_minorcorrect_idhandwriting_translate": """Please perform OCR on this scientific image and extract all of the words and text verbatim, but also correct any minor typos for scientific species names. If there is non-English text, translate all text to English, only returning the translated text. Identify any handwritten text. Do not explain your answer, only return the verbatim text:""",
            
            "default_plus_minorcorrect_Qwen": """Please perform OCR on this scientific image and extract all of the words and text verbatim. Then use your knowledge and the context of the surrounding text to also correct any minor typos caused by poor OCR for scientific species names. Do not explain your answer, only return the verbatim text:""",
            
            "default_plus_minorcorrect": """Please perform OCR on this scientific image and extract all of the words and text verbatim, but also correct any minor typos for scientific species names. Do not explain your answer, only return the verbatim text:""",

            "default": """Please perform OCR on this scientific image and extract all of the words and text verbatim. Do not explain your answer, only return the verbatim text:""",

            "handwriting_only": """Identify and extract all handwritten text from this image. Do not include printed text.Only return the handwritten text verbatim:""",

            "species_only": """Extract all scientific species names from this image. Do not include any other text. If a scientific species name has been stricken, crossed out, or redacted then place three tildes on either side. If a scientific species name is handwritten then place three asterisks symbols on either side. Correct minor typos in the species names if needed. Return only the species names:""",

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
