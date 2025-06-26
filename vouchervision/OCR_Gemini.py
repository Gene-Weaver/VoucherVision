import os, random, time, requests
import tempfile
from google import genai
from google.genai import types
from PIL import Image
from OCR_resize_for_VLMs import resize_image_to_min_max_pixels
from OCR_Prompt_Catalog import OCRPromptCatalog
from general_utils import calculate_cost


'''
Does not need to be downsampled like the other APIs or local 
Updated to use new Google GenAI SDK with dynamic thinking enabled
'''

class OCRGeminiProVision:
    def __init__(self, api_key, model_name="gemini-2.5-flash", max_output_tokens=4096, temperature=1, top_p=0.95, top_k=None, seed=123456, do_resize_img=False):
        """
        Initialize the OCRGeminiProVision class with the provided API key and model name.
        """
        self.supports_thinking = [
            'gemini-2.5-flash',
            'gemini-2.5-pro',
            ]
        self.path_api_cost = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_cost', 'api_cost.yaml')
        self.api_key = api_key
        self.do_resize_img = do_resize_img
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        if model_name not in self.supports_thinking:
            self.generation_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                # top_k=top_k,
                max_output_tokens=max_output_tokens,
                # seed=seed,  
            )
        else:
            self.generation_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                # top_k=top_k,  
                max_output_tokens=max_output_tokens,
                # seed=seed,  
                thinking_config=types.ThinkingConfig(thinking_budget=-1)  # Enable dynamic thinking
            )


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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        
        return ""
        # raise Exception(f"Failed to complete {func.__name__} after {max_retries} attempts.")
    
    
    # def exponential_backoff(self, func, *args, **kwargs):
    #     """
    #     Exponential backoff for a given function.
    #     Returns the longest successful result if possible, else "".
    #     """
    #     max_retries = 3
    #     best_result = ""
        
    #     for attempt in range(max_retries):
    #         try:
    #             result = func(*args, **kwargs)
    #             if isinstance(result, str) and len(result.strip()) > len(best_result.strip()):
    #                 best_result = result
    #             elif hasattr(result, 'text') and len(result.text.strip()) > len(best_result.strip()):
    #                 best_result = result
    #             return result  # if you want to return on first success
    #         except Exception as e:
    #             wait_time = (2 ** attempt) + random.uniform(0, 1)
    #             print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {wait_time:.2f} seconds...")
    #             time.sleep(wait_time)

    #     if best_result != "":
    #         print(f"Failed after {max_retries} attempts, but returning longest partial result.")
    #         return best_result
    #     else:
    #         print(f"Failed after {max_retries} attempts with no valid response. Returning empty string.")
    #         return ""
    
    def download_image_from_url(self, image_url):
        """
        Download an image from a URL and save it to a temporary file.
        
        Args:
            image_url (str): URL of the image to download.
            
        Returns:
            str: Path to the temporary file containing the downloaded image.
        """
        def download():
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Create a temporary file to store the image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file_path = temp_file.name
                
                # For large images, stream the content
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                    
            return temp_file_path
        
        return self.exponential_backoff(download)


    def upload_to_gemini_with_backoff(self, image_source):
        """
        Upload an image file to Gemini with exponential backoff.
        
        :param image_source: Path to the image file or URL of the image.
        :return: Uploaded file object with URI.
        """
        def upload():
            temp_files = []  # Keep track of temporary files to clean up
            
            try:
                # Check if image_source is a URL
                if isinstance(image_source, str) and image_source.startswith(('http://', 'https://')):
                    # Download the image from the URL as BytesIO
                    image_data = self.download_image_from_url(image_source)
                    image = Image.open(image_data)
                else:
                    # Use the local file path
                    image = Image.open(image_source)
                
                # Process the image (resize if needed)
                if self.do_resize_img:
                    image = resize_image_to_min_max_pixels(image)
                
                # Save to a temporary file for upload
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_path = temp_file.name
                    temp_files.append(temp_path)
                    image.save(temp_path, format="JPEG")
                
                # Upload the file using new SDK
                file = self.client.files.upload(file=temp_path)
                return file
            
            finally:
                # Clean up temporary files
                for temp_path in temp_files:
                    try:
                        if os.path.exists(temp_path):
                            # Close any open file handles
                            image.close()
                            # Add a small delay to ensure file is not in use
                            time.sleep(0.1)
                            os.remove(temp_path)
                    except Exception as e:
                        print(f"Warning: Failed to remove temporary file {temp_path}: {e}")

        return self.exponential_backoff(upload)

    def generate_content_with_backoff(self, prompt, uploaded_file, generation_config):
        """
        Generate content with exponential backoff.
        
        :param prompt: The prompt for the LLM.
        :param uploaded_file: The uploaded file object.
        :return: The response from the LLM.
        """
        def generate():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, uploaded_file],
                config=generation_config
            )
            
            # NEW: Check if the response is suspiciously short
            # if not response.text or len(response.text.strip()) < 30:
            #     raise Exception(f"Short or empty response (length={len(response.text.strip())}), retrying...")
            
            return response

        
        return self.exponential_backoff(generate)

    def ocr_gemini(self, image_path, prompt=None, temperature=1, top_p=0.95, top_k=None, max_output_tokens=None, seed=123456):
        """temperature=1, top_p=0.95
        Transcribes the text in the image using the Gemini model.

        :param image_path: Path to the image file.
        :param prompt: Instruction for the transcription task.
        :return: Transcription result as plain text.
        """
        # Update generation config with provided parameters
        if temperature:
            self.generation_config.temperature = temperature
        if top_p:
            self.generation_config.top_p = top_p
        # if top_k:
        #     self.generation_config.top_k = top_k  # Still commented as in original
        if max_output_tokens:
            self.generation_config.max_output_tokens = max_output_tokens
        # self.generation_config.seed = seed  # Still commented as in original

        
        overall_cost_in = 0
        overall_cost_out = 0
        overall_total_cost = 0
        overall_tokens_in = 0
        overall_tokens_out = 0
        rates_in = 0
        rates_out = 0
        total_cost = 0
        overall_response = ""

        request_generation_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens or self.generation_config.max_output_tokens,
        )

        # try: 
        if prompt is not None:
            # keys = ["default_plus_minorcorrect_addressstricken_idhandwriting",]
            keys = ["verbatim_with_annotations",]

        else:
            # keys = ["default", "default_plus_minorcorrect", "default_plus_minorcorrect_idhandwriting", "handwriting_only", "species_only", "detailed_metadata"]
            # keys = ["default_plus_minorcorrect_idhandwriting", "default_plus_minorcorrect_idhandwriting_translate", "species_only",]

            # keys = ["default_plus_minorcorrect_excludestricken_idhandwriting", "species_only",]
            # keys = ["default_plus_minorcorrect_addressstricken_idhandwriting", "species_only",] # last prior to annotations
            keys = ["verbatim_with_annotations",] # last prior to annotations
            
            # keys = ["default_plus_minorcorrect_idhandwriting", "species_only",]
            # keys = ["default_plus_minorcorrect_idhandwriting",]
        
        prompts = OCRPromptCatalog().get_prompts_by_keys(keys)
        for key, prompt in zip(keys, prompts):
            
            # Upload the image to Gemini
            uploaded_file = self.upload_to_gemini_with_backoff(image_path)

            # Generate content directly without starting a chat session
            response = self.generate_content_with_backoff(prompt, uploaded_file, request_generation_config)

            try:
                tokens_in = response.usage_metadata.prompt_token_count
                tokens_out = response.usage_metadata.candidates_token_count

                default_cost = (0, 0, 0, 0, 0)
                total_cost = default_cost

                if self.model_name == 'gemini-1.5-pro':
                    total_cost = calculate_cost('GEMINI_1_5_PRO', self.path_api_cost, tokens_in, tokens_out)
                elif self.model_name == 'gemini-1.5-flash':
                    total_cost = calculate_cost('GEMINI_1_5_FLASH', self.path_api_cost, tokens_in, tokens_out)
                elif self.model_name == 'gemini-1.5-flash-8b':
                    total_cost = calculate_cost('GEMINI_1_5_FLASH_8B', self.path_api_cost, tokens_in, tokens_out)                        
                elif self.model_name == 'gemini-2.0-flash-exp':
                    total_cost = calculate_cost('GEMINI_2_0_FLASH', self.path_api_cost, tokens_in, tokens_out)   
                elif self.model_name == 'gemini-2.0-flash':
                    total_cost = calculate_cost('GEMINI_2_0_FLASH', self.path_api_cost, tokens_in, tokens_out) 
                elif 'gemini-2.5-flash' in self.model_name:
                    total_cost = calculate_cost('GEMINI_2_5_FLASH', self.path_api_cost, tokens_in, tokens_out)   
                elif 'gemini-2.5-pro' in self.model_name:
                    total_cost = calculate_cost('GEMINI_2_5_PRO', self.path_api_cost, tokens_in, tokens_out)   

                cost_in, cost_out, total_cost, rates_in, rates_out = total_cost
                overall_cost_in += cost_in
                overall_cost_out += cost_out
                overall_total_cost += total_cost
                overall_tokens_in += tokens_in
                overall_tokens_out += tokens_out

                parsed_answer = response.text
                # if key == "species_only":
                #     parsed_answer = f"Based on context, determine which of these scientific names is the primary name: {parsed_answer}"

                if len(keys) > 1:
                    overall_response += (parsed_answer + "\n\n")
                else:
                    overall_response = parsed_answer
            except Exception as e:
                print(f"OCR failed: {e}")
        # finally:  # Use a finally block to *guarantee* deletion
        #     if uploaded_file.uri: # Check to ensure file was uploaded
        #         self.delete_gcs_file(uploaded_file.uri['uri'])

        try:
            return overall_response, overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out, overall_tokens_in, overall_tokens_out
        except Exception as e:
            print(f"overall_response failed: {e}")
            return "", overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out, overall_tokens_in, overall_tokens_out


# Example usage
if __name__ == "__main__":
    run_sweep = False
    API_KEY = "" #os.environ.get("GOOGLE_PALM_API")  # Replace with your actual API key
    
    image_paths = ["D:/D_Desktop/temp_50_2/ASC_3091190300_Asteraceae_Bidens_cernua.jpg",
        "D:/D_Desktop/temp_50_2/ASC_3091244339_Polygonaceae_Rumex_crispus.jpg",
        "D:/D_Desktop/temp_50_2/ASC_3091215365_Martyniaceae_Proboscidea_parviflora.jpg",]


    # image_path = "D:/Dropbox/VoucherVision/demo/demo_images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg"  # Replace with your image file path
    # image_path = 'C:/Users/willwe/Downloads/test_2024_12_04__13-49-56/Original_Images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg'
    
    ocr_tool = OCRGeminiProVision(api_key=API_KEY, model_name="gemini-2.5-pro")

    for i, image_path in enumerate(image_paths):
        print(f"WORKING ON [{i}]")
        response, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr_tool.ocr_gemini(image_path, temperature=1, top_k=1, top_p=0.95)
        print(response)


    # Revisit seed, https://github.com/google-gemini/generative-ai-python/issues/605
    # as of 12/4/2024 it was not available in the SDK yet
    if run_sweep:
        import csv

        success_counts = {}
        fail_counts = {}

        # Transcribe text from the image
        # reps = [0,1,2,]
        temps = [0, 0.5, 1, 1.5,]
        ks = [1, 3, 10, 40]
        ps = [0, 0.8, 0.5, 0.3]
        reps = [0,]
        # temps = [0, ]
        # ks = [1, ]
        # ps = [0, ]
        

        for rep in reps:

            for t in temps:
                for k in ks:
                    for p in ps:
                        response, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr_tool.ocr_gemini(image_path, temperature=t, top_k=k, top_p=p, seed=123456)
                        # print("Transcription Result:\n", response)

                        # Define the parameter tuple for tracking
                        param_set = (t, k, p, response)

                        if "jouvea pilosa" in response.lower():
                            # Increment success count
                            if param_set in success_counts:
                                success_counts[param_set] += 1
                            else:
                                success_counts[param_set] = 1
                            print("<<<< SUCCESS >>>>")
                            print(f"t {t}, k {k}, p {p}")
                            # print("Transcription Result:\n", response)
                        else:
                            # Increment failure count
                            if param_set in fail_counts:
                                fail_counts[param_set] += 1
                            else:
                                fail_counts[param_set] = 1
                            print("                     <<<< FAIL >>>>")
                            print(f"                     t {t}, k {k}, p {p}")
        # Display the results
        print("Success counts:", success_counts)
        print("Fail counts:", fail_counts)

        # Save to CSV
        with open('Gemini2flashexp_OCR_parameter_sweep_results_noSpecies.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write headers
            writer.writerow(['Temperature', 'Top K', 'Top P', 'Success Count', 'Fail Count', 'Response'])
            
            # Extract all unique parameter sets
            all_params = set(success_counts.keys()).union(set(fail_counts.keys()))
            
            # Write data rows
            for params in all_params:
                t, k, p, response = params
                success = success_counts.get(params, 0)
                fail = fail_counts.get(params, 0)
                writer.writerow([t, k, p, success, fail, response])