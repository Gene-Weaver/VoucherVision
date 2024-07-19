import os, base64, requests, yaml
from PIL import Image
from openai import OpenAI

from general_utils import calculate_cost

# PROMPT = """Please perform OCR on this scientific image and extract the printed and handwritten text verbatim. Do not explain your answer, only return the verbatim text in this JSON dictionary format: {'printed_text': '', 'handwritten_text': ''}"""
PROMPT = """Please perform OCR on this scientific image and extract all of the words and text verbatim. Do not explain your answer, only return the verbatim text:"""

class GPT4oMiniOCR:
    def __init__(self, api_key):
        self.api_key = api_key
        self.path_api_cost = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_cost', 'api_cost.yaml')


    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def ocr_gpt4o(self, image_path, resolution="low", max_tokens=512):
        # Getting the base64 string
        base64_image = self.encode_image(image_path)

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": PROMPT,
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": resolution,
                    }
                    }
                ]
                }
            ],
            "max_tokens": max_tokens
            }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json  = response.json()

        if "choices" in response_json :
            parsed_answer = response_json["choices"][0]["message"]["content"]
        else:
            parsed_answer = None

        usage_report = response_json.get('usage', {})
        tokens_in = usage_report["prompt_tokens"]
        tokens_out = usage_report["completion_tokens"]

        total_cost = calculate_cost('GPT_4o_mini_2024_07_18', self.path_api_cost, tokens_in, tokens_out)
        cost_in, cost_out, total_cost, rates_in, rates_out = total_cost

        return parsed_answer, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out




def main():
    # img_path = '/home/brlab/Downloads/gem_2024_06_26__02-26-02/Cropped_Images/By_Class/label/1.jpg'
    img_path = 'D:/D_Desktop/BR_1839468565_Ochnaceae_Campylospermum_reticulatum_label.jpg'
    
    # $env:OPENAI_API_KEY="KEY"
    API_KEY = ""

    
    ocr = GPT4oMiniOCR(API_KEY)
    
    parsed_answer, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr.ocr_gpt4o(img_path, resolution="low", max_tokens=512)
    print(f"Parsed Answer: {parsed_answer}")
    print(f"Total Cost: {total_cost}")
    
    parsed_answer, cost_in, cost_out, total_cost, rates_in, rates_out, tokens_in, tokens_out = ocr.ocr_gpt4o(img_path, resolution="high", max_tokens=512)
    print(f"Parsed Answer: {parsed_answer}")
    print(f"Total Cost: {total_cost}")

    


if __name__ == '__main__':
    main()