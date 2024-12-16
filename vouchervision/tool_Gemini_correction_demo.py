import os, random, time
import tempfile
# import google.generativeai as genai
from tool_Gemini_correction_catalog import GeminiCorrectionCatalog
from general_utils import calculate_cost
from google.ai.generativelanguage_v1beta.types import content

from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

'''
Does not need to be downsampled like the other APIs or local 
https://ai.google.dev/gemini-api/docs/vision?lang=python
'''

class ToolCorrection:
    def __init__(self, api_key, model_name="gemini-2.0-flash-exp", max_output_tokens=8192, temperature=1, top_p=0.95, top_k=40):
        """
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        Initialize the OCRGeminiProVision class with the provided API key and model name.
        """
        self.path_api_cost = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_cost', 'api_cost.yaml')
        self.api_key = api_key
        self.client = genai.Client(
            api_key=self.api_key,
        )
        # genai.configure(api_key=self.api_key)
        self.model_name = model_name
        # self.generation_config = {
        #     "temperature": temperature,
        #     "top_p": top_p,
        #     "top_k": top_k,
        #     "max_output_tokens": max_output_tokens,
        #     "response_mime_type": "text/plain",
        #     # "seed": seed,
        # }

        # self.model = genai.GenerativeModel(
        #     model_name=self.model_name,
        #     generation_config=self.generation_config,
        #     tools={"google_search": {}},
        # )

    def exponential_backoff(self, func, *args, **kwargs):
        """
        Exponential backoff for a given function.
        
        Args:
            func (function): The function to retry.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        
        Returns:
            The result of the function if successful.
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        
        raise Exception(f"Failed to complete {func.__name__} after {max_retries} attempts.")

    def generate_content_with_backoff(self, prompt):
        """
        Generate content with exponential backoff.
        
        :param prompt: The prompt for the LLM.
        :param uploaded_file: The uploaded file object.
        :return: The response from the LLM.
        """
        
        def generate():
            google_search_tool = Tool(
                google_search = GoogleSearch()
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                    temperature=self.temperature,
                )
            )
            return response
            # response = self.model.generate_content(
            #     [prompt],
            #     generation_config=self.generation_config
            # )
            # return response
        
        return self.exponential_backoff(generate)

    def make_correction(self, prompt=None, verbose=False, JSON="", temperature=None, top_p=None, top_k=None, max_output_tokens=None, seed=123456):
        """
        Transcribes the text in the image using the Gemini model.

        :param image_path: Path to the image file.
        :param prompt: Instruction for the transcription task.
        :return: Transcription result as plain text.
        """
        if temperature:
            self.temperature = temperature
        # if top_p:
        #     self.generation_config["top_p"] = top_p
        # if top_k:
        #     self.generation_config["top_k"] = top_k
        # if max_output_tokens:
        #     self.generation_config["max_output_tokens"] = max_output_tokens

        if verbose:
            verbose_version = "verbose_logic"
        else:
            verbose_version = "hide_logic"

        
        overall_cost_in = 0
        overall_cost_out = 0
        overall_total_cost = 0
        overall_tokens_in = 0
        overall_tokens_out = 0
        overall_response = ""


        objective = GeminiCorrectionCatalog().get_prompt(prompt)

        verbosity = GeminiCorrectionCatalog().get_prompt(verbose_version)

        json_add = GeminiCorrectionCatalog().get_prompt(verbose_version)

        constructed_prompt = objective + "\n\n" + verbosity + "\n\n" + json_add + "\n" + JSON
        
        # Upload the image to Gemini
        # Generate content directly without starting a chat session
        response = self.generate_content_with_backoff(constructed_prompt)
        print(response.candidates[0].grounding_metadata)

        try:
            tokens_in = response.usage_metadata.prompt_token_count
            tokens_out = response.usage_metadata.candidates_token_count

            default_cost = (0, 0, 0, 0, 0)
            total_cost = default_cost

            if self.model_name == 'gemini-1.5-pro':
                total_cost = calculate_cost('GEMINI_1_5_PRO', self.path_api_cost, tokens_in, tokens_out)
                cost_in, cost_out, total_cost, rates_in, rates_out = total_cost
                overall_cost_in += cost_in
                overall_cost_out += cost_out
                overall_total_cost += total_cost
                overall_tokens_in += tokens_in
                overall_tokens_out += tokens_out
            elif self.model_name == 'gemini-1.5-flash':
                total_cost = calculate_cost('GEMINI_1_5_FLASH', self.path_api_cost, tokens_in, tokens_out)
                cost_in, cost_out, total_cost, rates_in, rates_out = total_cost
                overall_cost_in += cost_in
                overall_cost_out += cost_out
                overall_total_cost += total_cost
                overall_tokens_in += tokens_in
                overall_tokens_out += tokens_out
            elif self.model_name == 'gemini-1.5-flash-8b':
                total_cost = calculate_cost('GEMINI_1_5_FLASH_8B', self.path_api_cost, tokens_in, tokens_out)   
                cost_in, cost_out, total_cost, rates_in, rates_out = total_cost
                overall_cost_in += cost_in
                overall_cost_out += cost_out
                overall_total_cost += total_cost
                overall_tokens_in += tokens_in
                overall_tokens_out += tokens_out
            else:                     
                overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out = total_cost
                overall_tokens_in = tokens_in
                overall_tokens_out = tokens_out


            overall_response = response.text
            
        except Exception as e:
            print(f"Correction failed: {e}")

        try:
            return overall_response, overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out, overall_tokens_in, overall_tokens_out
        except:
            return "", overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out, overall_tokens_in, overall_tokens_out


# Example usage
if __name__ == "__main__":
    API_KEY = "" #os.environ.get("GOOGLE_PALM_API")  # Replace with your actual API key
    
    JSON = """{
"filename": "20240125_233719",
"catalogNumber": "",
"scientificName": "Inetum microcarpum",
"genus": "Inetum",
"specificEpithet": "microcarpum",
"scientificNameAuthorship": "",
"collectedBy": "H. H. BARTLETT",
"collectorNumber": "6888",
"identifiedBy": "",
"identifiedDate": "",
"identifiedConfidence": "",
"identifiedRemarks": "",
"identificationHistory": "I. microcarpum Bl.",
"verbatimCollectionDate": "13 March 1927",
"collectionDate": "1927-03-13",
"collectionDateEnd": "",
"habitat": "Old jungle and second growth",
"specimenDescription": "",
"cultivated": "",
"continent": "Asia",
"country": "Indonesia",
"stateProvince": "Sumatra",
"county": "",
"locality": "near Aek Sordang, Loendoet\nConcession, Koealoe",
"verbatimCoordinates": "",
"decimalLatitude": "",
"decimalLongitude": "",
"minimumElevationInMeters": "",
"maximumElevationInMeters": "",
"elevationUnits": "",
"additionalText": "PLANTS OF SUMATRA (EAST COAST)\nCOLLECTED UNDER THE AUSPICES OF THE UNIVERSITY OF MICHIGAN\nAND THE SMITHSONIAN INSTITUTION"
}"""
    print(f"ORIGINAL\n{JSON}\n\n")
    
    Tool_Correction = ToolCorrection(api_key=API_KEY, model_name="gemini-2.0-flash-exp")

    # response, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = Tool_Correction.make_correction(prompt="fix_capitalization", verbose=False, JSON=JSON, temperature=1)
    # print(f"CAPITALIZATION\n{response}\n\n")
    response, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = Tool_Correction.make_correction(prompt="fix_taxonomy", verbose=True, JSON=JSON, temperature=1)
    print(f"TAXONOMY\n{response}\n\n")
    response, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = Tool_Correction.make_correction(prompt="grounded_location", verbose=False, JSON=JSON, temperature=1)
    print(f"GROUNDED LOCATION\n{response}\n\n")

