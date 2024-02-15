import os, time, json
# import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.generative_models._generative_models import HarmCategory, HarmBlockThreshold
from vertexai.language_models import TextGenerationModel
# from vertexai.preview.generative_models import GenerativeModel
from langchain.output_parsers import RetryWithErrorOutputParser
# from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI

from vouchervision.utils_LLM import SystemLoadMonitor, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template
from vouchervision.utils_taxonomy_WFO import validate_taxonomy_WFO
from vouchervision.utils_geolocate_HERE import validate_coordinates_here
from vouchervision.tool_wikipedia import validate_wikipedia

#https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk
#pip install --upgrade google-cloud-aiplatform
# from google.cloud import aiplatform

#### have to authenticate gcloud 
# gcloud auth login
# gcloud config set project XXXXXXXXX
# https://cloud.google.com/docs/authentication

class GooglePalm2Handler: 

    RETRY_DELAY = 10  # Wait 10 seconds before retrying
    MAX_RETRIES = 3  # Maximum number of retries
    TOKENIZER_NAME = 'gpt-4'
    VENDOR = 'google'
    STARTING_TEMP = 0.5

    def __init__(self, cfg, logger, model_name, JSON_dict_structure):
        self.cfg = cfg
        self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
        self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
        self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']

        self.logger = logger
        self.model_name = model_name
        self.JSON_dict_structure = JSON_dict_structure

        self.starting_temp = float(self.STARTING_TEMP)
        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp   

        self.monitor = SystemLoadMonitor(logger)

        self.parser = JsonOutputParser()

        # Define the prompt template
        self.prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self._set_config()

    def _set_config(self):
        # vertexai.init(project=os.environ['PALM_PROJECT_ID'], location=os.environ['PALM_LOCATION'])
        self.config = {
                "max_output_tokens": 1024,
                "temperature": self.starting_temp,
                "top_p": 1.0,
                "top_k": 40,
            }
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        self._build_model_chain_parser()

    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        self.json_report.set_text(text_main=f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.adjust_temp += self.temp_increment
        self.config['temperature'] = self.adjust_temp    

    def _reset_config(self):
        self.json_report.set_text(text_main=f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.adjust_temp = self.starting_temp
        self.config['temperature'] = self.starting_temp   

    def _build_model_chain_parser(self):
        # Instantiate the parser and the retry parser
        # self.llm_model = ChatGoogleGenerativeAI(model=self.model_name)
        self.llm_model = VertexAI(model=self.model_name)
        
        self.retry_parser = RetryWithErrorOutputParser.from_llm(
                                                parser=self.parser,
                                                llm=self.llm_model,
                                                max_retries=self.MAX_RETRIES)
        # Prepare the chain
        self.chain = self.prompt | self.call_google_palm2

    # Define a function to format the input for Google PaLM call
    # https://cloud.google.com/vertex-ai/docs/generative-ai/migrate/migrate-palm-to-gemini?_ga=2.225326234.-1652490527.1705461451&_gac=1.186295771.1706291573.CjwKCAiAzc2tBhA6EiwArv-i6QCpx7xTP0yrBy9KKSwno3QXOWUe14mbp9RGZO0ShcbtFqyXii2PnRoCywgQAvD_BwE
    def call_google_palm2(self, prompt_text):
        model = TextGenerationModel.from_pretrained(self.model_name)
        response = model.predict(prompt_text.text,
                                max_output_tokens=self.config.get('max_output_tokens'),
                                temperature=self.config.get('temperature'),
                                top_p=self.config.get('top_p'))
        # model = GenerativeModel(self.model_name)

        # response = model.generate_content(prompt_text.text,generation_config=self.config, safety_settings=self.safety_settings, stream=False)
        return response.text


    def call_llm_api_GooglePalm2(self, prompt_template, json_report, paths):
        _____, ____, _, __, ___, json_file_path_wiki, txt_file_path_ind_prompt = paths
        self.json_report = json_report
        self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        self.monitor.start_monitoring_usage()
        nt_in = 0
        nt_out = 0

        ind = 0
        while ind < self.MAX_RETRIES:
            ind += 1
            try:
                model_kwargs = {"temperature": self.adjust_temp}
                # Invoke the chain to generate prompt text
                response = self.chain.invoke({"query": prompt_template, "model_kwargs": model_kwargs})

                # Use retry_parser to parse the response with retry logic
                output = self.retry_parser.parse_with_prompt(response, prompt_value=prompt_template)

                if output is None:
                    self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{response}')
                    self._adjust_config()
                else:
                    nt_in = count_tokens(prompt_template, self.VENDOR, self.TOKENIZER_NAME)
                    nt_out = count_tokens(response, self.VENDOR, self.TOKENIZER_NAME)
                    
                    output = validate_and_align_JSON_keys_with_template(output, self.JSON_dict_structure)
                    if output is None:
                        self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{response}')
                        self._adjust_config()           
                    else:
                        self.monitor.stop_inference_timer() # Starts tool timer too

                        json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                        output, WFO_record = validate_taxonomy_WFO(self.tool_WFO, output, replace_if_success_wfo=False) 
                        output, GEO_record = validate_coordinates_here(self.tool_GEO, output, replace_if_success_geo=False) 
                        validate_wikipedia(self.tool_wikipedia, json_file_path_wiki, output)

                        save_individual_prompt(sanitize_prompt(prompt_template), txt_file_path_ind_prompt)

                        self.logger.info(f"Formatted JSON:\n{json.dumps(output,indent=4)}")
                        
                        usage_report = self.monitor.stop_monitoring_report_usage()    

                        if self.adjust_temp != self.starting_temp:            
                            self._reset_config()

                        json_report.set_text(text_main=f'LLM call successful')
                        return output, nt_in, nt_out, WFO_record, GEO_record, usage_report

            except Exception as e:
                self.logger.error(f'{e}')
                
                self._adjust_config()           
                time.sleep(self.RETRY_DELAY)

        self.logger.info(f"Failed to extract valid JSON after [{ind}] attempts")
        self.json_report.set_text(text_main=f'Failed to extract valid JSON after [{ind}] attempts')

        usage_report = self.monitor.stop_monitoring_report_usage()                
        self._reset_config()

        json_report.set_text(text_main=f'LLM call failed')
        return None, nt_in, nt_out, None, None, usage_report