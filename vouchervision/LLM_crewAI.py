import time, os, json
import torch
from crewai import Agent, Task, Crew, Process
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_classic.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import RetryWithErrorOutputParser

class VoucherVisionWorkflow:
    MODEL = 'gpt-4o'
    SHARED_INSTRUCTIONS = """
    instructions: 
    1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below.
    2. Map the unstructured OCR text to the appropriate JSON key and populate the field given the user-defined rules.
    3. JSON key values are permitted to remain empty strings if the corresponding information is not found in the unstructured OCR text.
    4. Duplicate dictionary fields are not allowed.
    5. Ensure all JSON keys are in camel case.
    6. Ensure new JSON field values follow sentence case capitalization.
    7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template.
    8. Ensure output JSON string is valid JSON format. It should not have trailing commas or unquoted keys.
    9. Only return a JSON dictionary represented as a string. You should not explain your answer.

    JSON structure: 
    {"catalogNumber": "", "scientificName": "", "genus": "", "specificEpithet": "", "speciesNameAuthorship": "", "collectedBy": "", "collectorNumber": "", "identifiedBy": "", "verbatimCollectionDate": "", "collectionDate": "", "collectionDateEnd": "", "occurrenceRemarks": "", "habitat": "", "cultivated": "", "country": "", "stateProvince": "", "county": "", "locality": "", "verbatimCoordinates": "", "decimalLatitude": "", "decimalLongitude": "", "minimumElevationInMeters": "", "maximumElevationInMeters": "", "elevationUnits": ""}
    """

    EXPECTED_OUTPUT_STRUCTURE = """{
        "JSON_OUTPUT": {
            "catalogNumber": "", "scientificName": "", "genus": "", "specificEpithet": "", 
            "speciesNameAuthorship": "", "collectedBy": "", "collectorNumber": "", 
            "identifiedBy": "", "verbatimCollectionDate": "", "collectionDate": "", 
            "collectionDateEnd": "", "occurrenceRemarks": "", "habitat": "", "cultivated": "", 
            "country": "", "stateProvince": "", "county": "", "locality": "", 
            "verbatimCoordinates": "", "decimalLatitude": "", "decimalLongitude": "", 
            "minimumElevationInMeters": "", "maximumElevationInMeters": "", "elevationUnits": ""
        },
        "explanation": ""
    }"""

    def __init__(self, api_key, librarian_knowledge_path):
        self.api_key = api_key
        os.environ['OPENAI_API_KEY'] = self.api_key

        self.librarian_knowledge = self.load_librarian_knowledge(librarian_knowledge_path)
        self.worker_agent = self.create_worker_agent()
        self.supervisor_agent = self.create_supervisor_agent()

    def load_librarian_knowledge(self, path):
        with open(path) as f:
            return json.load(f)

    def query_librarian(self, guideline_field):
        print(f"query_librarian: {guideline_field}")
        return self.librarian_knowledge.get(guideline_field, "Guideline not found.")

    def create_worker_agent(self):
        return Agent(
            role="Transcriber and JSON Formatter",
            goal="Transcribe product labels accurately and format them into a structured JSON dictionary. Only return a JSON dictionary.",
            backstory="You're an AI trained to transcribe product labels and format them into JSON.",
            verbose=True,
            allow_delegation=False,
            llm=ChatOpenAI(model=self.MODEL, openai_api_key=self.api_key),
            prompt_instructions=self.SHARED_INSTRUCTIONS
        )

    def create_supervisor_agent(self):
        class SupervisorAgent(Agent):
            def correct_with_librarian(self, workflow, transcription, json_dict, guideline_field):
                guideline = workflow.query_librarian(guideline_field)
                corrected_transcription = self.correct(transcription, guideline)
                corrected_json = self.correct_json(json_dict, guideline)
                explanation = f"Corrected {json_dict} based on guideline {guideline_field}: {guideline}"
                return corrected_transcription, {"JSON_OUTPUT": corrected_json, "explanation": explanation}

        return SupervisorAgent(
            role="Corrector",
            goal="Ensure accurate transcriptions and JSON formatting according to specific guidelines. Compare the OCR text to the JSON dictionary and make any required corrections. Given your knowledge, make sure that the values in the JSON object make sense given the cumulative context of the OCR text. If you correct the provided JSON, then state the corrections. Otherwise say that the original worker was correct.",
            backstory="You're an AI trained to correct transcriptions and JSON formatting, consulting the librarian for guidance.",
            verbose=True,
            allow_delegation=False,
            llm=ChatOpenAI(model=self.MODEL, openai_api_key=self.api_key),
            prompt_instructions=self.SHARED_INSTRUCTIONS
        )

    def extract_json_from_string(self, input_string):
        json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')
        match = json_pattern.search(input_string)
        if match:
            return match.group(0)
        return None

    def extract_json_via_api(self, text):
        self.api_key = self.api_key
        extraction_prompt = f"I only need the JSON inside this text. Please return only the JSON object.\n\n{text}"
        response = openai.ChatCompletion.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": extraction_prompt}
            ]
        )
        return self.extract_json_from_string(response['choices'][0]['message']['content'])


    def run_workflow(self, ocr_text):
        openai_model = ChatOpenAI(api_key=self.api_key, model=self.MODEL)

        self.worker_agent.llm = openai_model
        self.supervisor_agent.llm = openai_model

        transcription_and_formatting_task = Task(
            description=f"Transcribe product label and format into JSON. OCR text: {ocr_text}", 
            agent=self.worker_agent,
            inputs={"ocr_text": ocr_text},
            expected_output=self.EXPECTED_OUTPUT_STRUCTURE
        )

        crew = Crew(
            agents=[self.worker_agent],
            tasks=[transcription_and_formatting_task],
            verbose=True,
            process=Process.sequential,
        )

        # Run the transcription and formatting task
        transcription_and_formatting_result = transcription_and_formatting_task.execute()
        print("Worker Output JSON:", transcription_and_formatting_result)

        # Pass the worker's JSON output to the supervisor for correction
        correction_task = Task(
            description=f"Correct transcription and JSON format. OCR text: {ocr_text}", 
            agent=self.supervisor_agent,
            inputs={"ocr_text": ocr_text, "json_dict": transcription_and_formatting_result},
            expected_output=self.EXPECTED_OUTPUT_STRUCTURE,
            workflow=self  # Pass the workflow instance to the task
        )

        correction_result = correction_task.execute()
        
        try:
            corrected_json_with_explanation = json.loads(correction_result)
        except json.JSONDecodeError:
            # If initial parsing fails, make a call to OpenAI to extract only the JSON
            corrected_json_string = self.extract_json_via_api(correction_result)
            if not corrected_json_string:
                raise ValueError("No JSON found in the supervisor's output.")
            corrected_json_with_explanation = json.loads(corrected_json_string)

        corrected_json = corrected_json_with_explanation["JSON_OUTPUT"]
        explanation = corrected_json_with_explanation["explanation"]

        print("Supervisor Corrected JSON:", corrected_json)
        print("\nCorrection Explanation:", explanation)

        return corrected_json, explanation

if __name__ == "__main__":
    api_key = ""
    librarian_knowledge_path = "D:/Dropbox/VoucherVision/vouchervision/librarian_knowledge.json"
    
    ocr_text  = "HERBARIUM OF MARYGROVE COLLEGE Name Carex scoparia V. condensa Fernald Locality Interlaken , Ind . Date 7/20/25 No ... ! Gerould Wilhelm & Laura Rericha \" Interlaken , \" was the site for many years of St. Joseph Novitiate , run by the Brothers of the Holy Cross . The buildings were on the west shore of Silver Lake , about 2 miles NE of Rolling Prairie , LaPorte Co. Indiana , ca. 41.688 \u00b0 N , 86.601 \u00b0 W Collector : Sister M. Vincent de Paul McGivney February 1 , 2011 THE UNIVERS Examined for the Flora of the Chicago Region OF 1817 MICH ! Ciscoparia SMVdeP University of Michigan Herbarium 1386297 copyright reserved cm Collector wortet 2010"
    workflow = VoucherVisionWorkflow(api_key, librarian_knowledge_path)
    workflow.run_workflow(ocr_text)

    ocr_text  = "CM 1 2 3 QUE : Mt.Jac.Cartier Parc de la Gasp\u00e9sie 13 Aug1988 CA Vogt on Solidago MCZ - ENT OXO Bombus vagans Smith ' det C.A. Vogt 1988 UIUC USDA BBDP 021159 00817079 "
    workflow = VoucherVisionWorkflow(api_key, librarian_knowledge_path)
    workflow.run_workflow(ocr_text)

    ocr_text  = "500 200 600 300 dots per inch ( optical ) 700 400 800 500 850 550 850 550 Golden Thread inches centimeters 500 200 600 300 dots per inch ( optical ) 11116 L * 39.12 65.43 49.87 44.26 b * 15.07 18.72 -22.29 22.85 4 -4.34 -13.80 3 13.24 18.11 2 1 5 9 7 11 ( A ) 10 -0.40 48.55 55.56 70.82 63.51 39.92 52.24 97.06 92.02 9.82 -33.43 34.26 11.81 -24.49 -0.35 59.60 -46.07 18.51 8 6 12 13 14 15 87.34 82.14 72.06 62.15 09.0- -0.75 -1.06 -1.19 -1.07 1.13 0.23 0.21 0.43 0.28 0.19 800 500 D50 Illuminant , 2 degree observer Density 0.04 0.09 0.15 0.22 Fam . Saurauiaceae J. G. Agardh Saurauia nepaulensis DC . S. Vietnam , Prov . Kontum . NW slopes of Ngoc Linh mountain system at 1200 m alt . near Ngoc Linh village . Secondary marshland with grasses and shrubs . Tree up to 5 m high . Flowers light rosy - pink . No VH 007 0.36 0.51 23.02.1995 International Botanical Expedition of the U.S.A. National Geographic Society ( grant No 5094-93 ) Participants : L. Averyanov , N.T. Ban , N. Q. Binh , A. Budantzev , L. Budantzev , N.T. Hiep , D.D. Huyen , P.K. Loc , N.X. Tam , G. Yakovlev BOTANICAL RESEARCH INSTITUTE OF TEXAS BRIT610199 Botanical Research Institute of Texas IMAGED 08 JUN 2021 FLORA OF VIETNAM "
    workflow = VoucherVisionWorkflow(api_key, librarian_knowledge_path)
    workflow.run_workflow(ocr_text)

    ocr_text  = "Russian - Vietnamese Tropical Centre Styrax argentifolius H.L. Li SOUTHERN VIETNAM Dak Lak prov . , Lak distr . , Bong Krang municip . Chu Yang Sin National Park 10 km S from Krong Kmar village River bank N 12 \u00b0 25 ' 24 \" E 108 \u00b0 21 ' 04 \" elev . 900 m Nuraliev M.S. No 1004 part of MW 0750340 29.05.2014 Materials of complex expedition in spring 2014 BOTANICAL RESEARCH INSTITUTE OF TEXAS ( BRIT ) Styrax benzoides Craib Det . by Peter W. Fritsch , September 2017 0 1 2 3 4 5 6 7 8 9 10 BOTANICAL RESEARCH INSTITUTE OF TEXAS BOTANICAL IMAGED RESEARCH INSTITUTE OF 10 JAN 2013 BRIT402114 copyright reserved cm BOTANICAL RESEARCH INSTITUTE OF TEXAS TM P CameraTrax.com BRIT . TEXAS "
    workflow = VoucherVisionWorkflow(api_key, librarian_knowledge_path)
    workflow.run_workflow(ocr_text)