from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Transcription(BaseModel):
    Transcription_Printed_Text: str = Field(description="The transcription of all printed text in the image.")
    Transcription_Handwritten_Text: str = Field(description="The transcription of all handwritten text in the image.")
    
class OCRGPT4VisionPreview:
    def __init__(self, logger, api_key, endpoint_url="https://gpt-4-vision-preview-api.com/ocr"):
        self.logger = logger
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.parser = JsonOutputParser(pydantic_object=Transcription)

    def transcribe_image(self, image_file):
        self.logger.start_monitoring_usage()

        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {'image': open(image_file, 'rb')}
        response = requests.post(self.endpoint_url, headers=headers, files=files)

        if response.status_code == 200:
            json_response = response.json()
            transcription = self.parser.parse(json_response)
        else:
            self.logger.log_error("Failed to transcribe image")
            transcription = {"Transcription": "Error"}

        usage_report = self.logger.stop_monitoring_report_usage()

        return transcription, usage_report
