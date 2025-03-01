import os, random, time
import tempfile
import google.generativeai as genai
from PIL import Image
from OCR_resize_for_VLMs import resize_image_to_min_max_pixels
from OCR_Prompt_Catalog import OCRPromptCatalog
from general_utils import calculate_cost


'''
Does not need to be downsampled like the other APIs or local 
https://ai.google.dev/gemini-api/docs/vision?lang=python
'''

class OCRGeminiProVision:
    def __init__(self, api_key, model_name="gemini-1.5-pro", max_output_tokens=1024, temperature=0.5, top_p=0.3, top_k=3, seed=123456, do_resize_img=False):
        """
        Initialize the OCRGeminiProVision class with the provided API key and model name.
        """
        self.path_api_cost = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_cost', 'api_cost.yaml')
        self.api_key = api_key
        self.do_resize_img = do_resize_img
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "text/plain",
            # "seed": seed,
        }
        self.model = genai.GenerativeModel(
            model_name=self.model_name, generation_config=self.generation_config
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


    def upload_to_gemini_with_backoff(self, image_path, mime_type="image/jpeg"):
        """
        Upload an image file to Gemini with exponential backoff.
        
        :param image_path: Path to the image file.
        :param mime_type: MIME type of the image.
        :return: Uploaded file object with URI.
        """
        genai.configure(api_key=self.api_key)

        def upload():
            if self.do_resize_img:
                image = Image.open(image_path)
                resized_image = resize_image_to_min_max_pixels(image)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    resized_image.save(temp_file.name, format="JPEG")
                    temp_file_path = temp_file.name

                file = genai.upload_file(temp_file_path, mime_type=mime_type)
                os.remove(temp_file_path)
            else:
                file = genai.upload_file(image_path, mime_type=mime_type)
            return file

        return self.exponential_backoff(upload)

    def generate_content_with_backoff(self, prompt, uploaded_file):
        """
        Generate content with exponential backoff.
        
        :param prompt: The prompt for the LLM.
        :param uploaded_file: The uploaded file object.
        :return: The response from the LLM.
        """
        def generate():
            response = self.model.generate_content(
                [prompt, uploaded_file],
                generation_config=self.generation_config
            )
            return response
        
        return self.exponential_backoff(generate)

    def ocr_gemini(self, image_path, prompt=None, temperature=None, top_p=None, top_k=None, max_output_tokens=None, seed=123456):
        """
        Transcribes the text in the image using the Gemini model.

        :param image_path: Path to the image file.
        :param prompt: Instruction for the transcription task.
        :return: Transcription result as plain text.
        """
        if temperature:
            self.generation_config["temperature"] = temperature
        if top_p:
            self.generation_config["top_p"] = top_p
        if top_k:
            self.generation_config["top_k"] = top_k
        if max_output_tokens:
            self.generation_config["max_output_tokens"] = max_output_tokens
        # self.generation_config["seed"] = seed

        
        overall_cost_in = 0
        overall_cost_out = 0
        overall_total_cost = 0
        overall_tokens_in = 0
        overall_tokens_out = 0
        overall_response = ""

        # try: 
        if prompt is None:
            # keys = ["default", "default_plus_minorcorrect", "default_plus_minorcorrect_idhandwriting", "handwriting_only", "species_only", "detailed_metadata"]
            # keys = ["default_plus_minorcorrect_idhandwriting", "default_plus_minorcorrect_idhandwriting_translate", "species_only",]
            keys = ["default_plus_minorcorrect_excludestricken_idhandwriting", "species_only",]
            # keys = ["default_plus_minorcorrect_idhandwriting", "species_only",]
            # keys = ["default_plus_minorcorrect_idhandwriting",]

            prompts = OCRPromptCatalog().get_prompts_by_keys(keys)
            for key, prompt in zip(keys, prompts):
                
                # Upload the image to Gemini
                uploaded_file = self.upload_to_gemini_with_backoff(image_path)

                # Generate content directly without starting a chat session
                response = self.generate_content_with_backoff(prompt, uploaded_file)

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
                    elif self.model_name == 'gemini-2.0-pro':
                        total_cost = calculate_cost('GEMINI_2_0_PRO', self.path_api_cost, tokens_in, tokens_out)   

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
        except:
            return "", overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out, overall_tokens_in, overall_tokens_out


# Example usage
if __name__ == "__main__":
    run_sweep = False
    API_KEY = "" #os.environ.get("GOOGLE_PALM_API")  # Replace with your actual API key
    
    image_paths = ["/Users/williamweaver/Desktop/translate/00515126.jpg",
        "/Users/williamweaver/Desktop/translate/04322236.jpg",
        "/Users/williamweaver/Desktop/translate/04357250.jpg",]


    # image_path = "D:/Dropbox/VoucherVision/demo/demo_images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg"  # Replace with your image file path
    # image_path = 'C:/Users/willwe/Downloads/test_2024_12_04__13-49-56/Original_Images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg'
    
    ocr_tool = OCRGeminiProVision(api_key=API_KEY, model_name="gemini-2.0-flash-exp")

    for image_path in image_paths:
        response, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr_tool.ocr_gemini(image_path, temperature=1, top_k=1, top_p=0)
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

