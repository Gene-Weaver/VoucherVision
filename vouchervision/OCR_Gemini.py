import os
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
    def __init__(self, api_key, model_name="gemini-1.5-pro", max_output_tokens=1024, temperature=1, top_p=0, top_k=1, do_resize_img=False):
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
        }
        self.model = genai.GenerativeModel(
            model_name=self.model_name, generation_config=self.generation_config
        )

    def upload_to_gemini(self, image_path, mime_type="image/jpeg"):
        
        """
        Upload an image file to Gemini.

        :param image_path: Path to the image file.
        :param mime_type: MIME type of the image.
        :return: Uploaded file object with URI.
        """
        genai.configure(api_key=self.api_key)

        if self.do_resize_img:
            image = Image.open(image_path)
            resized_image = resize_image_to_min_max_pixels(image)
            # Save the resized image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                resized_image.save(temp_file.name, format="JPEG")
                temp_file_path = temp_file.name

            # Upload the resized image to Gemini
            file = genai.upload_file(temp_file_path, mime_type=mime_type)
            print(f"Uploaded file '{file.display_name}' as: {file.uri}")
            os.remove(temp_file_path)
        else:
            file = genai.upload_file(image_path, mime_type=mime_type)
            print(f"Uploaded file '{image_path}'")

        return file

    def ocr_gemini(self, image_path, prompt=None):
        """
        Transcribes the text in the image using the Gemini model.

        :param image_path: Path to the image file.
        :param prompt: Instruction for the transcription task.
        :return: Transcription result as plain text.
        """
        overall_cost_in = 0
        overall_cost_out = 0
        overall_total_cost = 0
        overall_tokens_in = 0
        overall_tokens_out = 0
        overall_response = ""

        if prompt is None:
            # keys = ["default", "default_plus_minorcorrect", "default_plus_minorcorrect_idhandwriting", "handwriting_only", "species_only", "detailed_metadata"]
            keys = ["default_plus_minorcorrect_idhandwriting",]

            prompts = OCRPromptCatalog().get_prompts_by_keys(keys)
            for key, prompt in zip(keys, prompts):
                
                # Upload the image to Gemini
                uploaded_file = self.upload_to_gemini(image_path)

                # Generate content directly without starting a chat session
                response = self.model.generate_content(
                    [prompt, uploaded_file]
                )
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

                    cost_in, cost_out, total_cost, rates_in, rates_out = total_cost
                    overall_cost_in += cost_in
                    overall_cost_out += cost_out
                    overall_total_cost += total_cost
                    overall_tokens_in += tokens_in
                    overall_tokens_out += tokens_out
                    if len(keys) > 1:
                        overall_response += (response.text + "\n\n")
                    else:
                        overall_response = response.text
                except Exception as e:
                    print(f"OCR failed: {e}")

        try:
            return overall_response, overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out, overall_tokens_in, overall_tokens_out
        except:
            return "", overall_cost_in, overall_cost_out, overall_total_cost, rates_in, rates_out, overall_tokens_in, overall_tokens_out


# Example usage
if __name__ == "__main__":
    API_KEY = "" #os.environ.get("GOOGLE_PALM_API")  # Replace with your actual API key
    image_path = "D:/Dropbox/VoucherVision/demo/demo_images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg"  # Replace with your image file path
    ocr_tool = OCRGeminiProVision(api_key=API_KEY)

    # Transcribe text from the image
    response, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr_tool.ocr_gemini(image_path)
    print("Transcription Result:\n", response)
