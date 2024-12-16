class GeminiCorrectionCatalog:
    """
    A catalog of correction prompts for different versions and use cases.
    This class provides a central repository for storing and retrieving prompts.
    """
    def __init__(self):
        # Dictionary to store prompts by version
        self.prompts = {
            "fix_capitalization": """Given the JSON object below, look at each of the values and correct the capitalization. I need all entries to follow English rules of capitalization for all instances of proper nouns, names, taxonomy, and locations. Some fields may not be grammatically correct which is okay. Replace new lines in the values with spaces. None of the values should be all caps.""",
            
            "fix_taxonomy": """Given the JSON object below, search Google for “POWO” and the value in the genus and specificEpithet fields in the JSON object. These fields are achieved via OCR of cursive, so genus plus specificEpithet may require substitutions of characters to become a valid scientific name.If there is not an exact match and the genus plus specificEpithet is not the synonym for something else, then you need to perform an additional search and use your knowledge of scientific plant names to figure out why the name is wrong. Start with the genus, make sure it is a valid genus name. If not, use your knowledge of common OCR substitution errors to figure out the correct genus name, which should have roughly the same number of characters in the name. Then move to the specificEpithet by trying to search for the new genus name plus the original specificEpithet. If that is a valid scientific name, then that is the new scientific name. Otherwise, use your knowledge of common OCR substitution errors to figure out the correct specificEpithet name.""",  
            
            "grounded_location": """Given the JSON object below, search Google for “midwestherbaria”, the values in collectedBy, the year from collectionDate, country, stateProvince, county, and locality fields in the JSON object. Use this search data to help correct or fill in the information contained in the JSON object, since its location fields may not be as complete as those that are already in the database. You are essentially performing a georeferencing task. The queries have a different locality structure that you need to parse. It will include in order, the country, stateProvince, county, and then all other specific locality information or “no data” for some fields. If there is a strong match given the context, infer that the existing database is better and revise the existing JSON. Return the new JSON object, include one new field at the end called similarCollectionNumber which returns the collection number that was used to infer the changes. If none, return None for the similarCollectionNumber value.""",
            
            "show_JSON": """This is the JSON object:""",
            
            "verbose_logic": """Explain your process before finally returning the JSON object.""",
            
            "hide_logic": """Hide your thought process from the final output, only return the final JSON object.""",
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
