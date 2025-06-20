import os, time, json
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.generative_models._generative_models import HarmCategory, HarmBlockThreshold
from langchain.output_parsers import RetryWithErrorOutputParser
# from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI

from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template
from google import genai
from google.genai import types
from google.genai.types import Tool, GoogleSearch, GenerateContentConfig

class GoogleGeminiHandler: 

    RETRY_DELAY = 10  # Wait 10 seconds before retrying
    MAX_RETRIES = 3  # Maximum number of retries
    TOKENIZER_NAME = 'gpt-4'
    VENDOR = 'google'
    STARTING_TEMP = 1

    THINK_BUDGET = 2048

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation, exit_early_for_JSON=False, exit_early_with_WFO=False):
        self.exit_early_for_JSON = exit_early_for_JSON
        self.exit_early_with_WFO = exit_early_with_WFO

        self.cfg = cfg
        if self.exit_early_for_JSON and self.exit_early_with_WFO: # Add WFO to the output
            self.tool_WFO = True
            self.tool_GEO = False
            self.tool_wikipedia = False
        else:
            self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
            self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
            self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']

        try:
            self.tool_google = self.cfg['leafmachine']['project']['tool_google']
        except:
            self.tool_google = False

        self.logger = logger
        self.model_name = model_name
        self.JSON_dict_structure = JSON_dict_structure

        self.config_vals_for_permutation = config_vals_for_permutation
        
        self.monitor = SystemLoadMonitor(logger)

        self.parser = JsonOutputParser()

        self.google_search_tool = Tool(google_search = GoogleSearch())
        # self.google_search_tool = {'google_search': {}}

        # Define the prompt template
        self.prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self._set_config()


    def _set_config(self):
        # os.environ['GOOGLE_API_KEY'] # Must be set too for the retry call, set in VoucherVision class along with other API Keys
        # vertexai.init(project=os.environ['PALM_PROJECT_ID'], location=os.environ['PALM_LOCATION'])
        if self.config_vals_for_permutation:
            self.starting_temp = float(self.config_vals_for_permutation.get('google').get('temperature'))
            self.config = {
                    'max_output_tokens': self.config_vals_for_permutation.get('google').get('max_output_tokens'),
                    'temperature': self.starting_temp,
                    'top_p': self.config_vals_for_permutation.get('google').get('top_p'),
                    }
        else:
            self.starting_temp = float(self.STARTING_TEMP)
            self.config = {
                "max_output_tokens": 1024,
                "temperature": self.starting_temp,
                "top_p": 1.0,
            }

        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp   

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        self._build_model_chain_parser()

    def log_response_metrics(self, response):
        metrics = {
            'Search Query': None,
            'Search Pages': None,
            'Prompt Tokens': None,
            'Thoughts Tokens': [None, self.THINK_BUDGET],
            'Output Tokens': None,
            'Total Tokens': None
        }
        
        try: metrics['Search Query'] = response.candidates[0].grounding_metadata.web_search_queries
        except: pass
        try: metrics['Search Pages'] = ", ".join([site.web.title for site in response.candidates[0].grounding_metadata.grounding_chunks])
        except: pass
        try: metrics['Prompt Tokens'] = response.usage_metadata.prompt_token_count
        except: pass
        try: metrics['Thoughts Tokens'][0] = response.usage_metadata.thoughts_token_count
        except: pass
        try: metrics['Output Tokens'] = response.usage_metadata.candidates_token_count
        except: pass
        try: metrics['Total Tokens'] = response.usage_metadata.total_token_count
        except: pass
        
        for key, value in metrics.items():
            if key == 'Thoughts Tokens':
                self.logger.info(f'[GEMINI] {key}: {value[0]} / {value[1]}')
            else:
                self.logger.info(f'[GEMINI] {key}: {value}')
                
        return True
        
    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        if self.json_report:            
            self.json_report.set_text(text_main=f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.adjust_temp += self.temp_increment
        self.config['temperature'] = self.adjust_temp   

    def _reset_config(self):
        if self.json_report:            
            self.json_report.set_text(text_main=f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.adjust_temp = self.starting_temp
        self.config['temperature'] = self.starting_temp  

    def _build_model_chain_parser(self):
        # Instantiate the LLM class for Google Gemini
        if not self.exit_early_for_JSON:
            self.llm_model = ChatGoogleGenerativeAI(model=self.model_name, 
                                    max_output_tokens=self.config.get('max_output_tokens'),
                                    top_p=self.config.get('top_p'),
                                    temperature=self.config.get('temperature'),
                                    )    
        else: # For vvgo
            self.llm_model = ChatGoogleGenerativeAI(model=self.model_name, 
                                        max_output_tokens=self.config.get('max_output_tokens'),
                                        top_p=self.config.get('top_p'),
                                        temperature=self.config.get('temperature'),
                                        api_key=os.environ.get("API_KEY")
                                        )
        # self.llm_model = VertexAI(model='gemini-1.0-pro', 
        #                           max_output_tokens=self.config.get('max_output_tokens'),
        #                           top_p=self.config.get('top_p'))   

        # Set up the retry parser with the runnable
        self.retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=self.parser, 
            llm=self.llm_model, 
            max_retries=self.MAX_RETRIES)
        # Prepare the chain
        self.chain = self.prompt | self.call_google_gemini     

    # Define a function to format the input for Google Gemini call
    def call_google_gemini(self, prompt_text):
        if ("2.5" in self.model_name):# or ("2.0" in self.model_name):
            try:
                try:
                    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
                except:
                    client = genai.Client(api_key=os.environ.get("API_KEY"))

                if self.tool_google:
                    self.logger.info(f'[GEMINI] {self.model_name} --- THINK[{self.THINK_BUDGET}] --- TOOLS[GOOGLE SEARCH]')
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_text.text,
                        config=types.GenerateContentConfig(
                            tools=[self.google_search_tool],
                            thinking_config=types.ThinkingConfig(thinking_budget=self.THINK_BUDGET),
                            response_modalities=["TEXT"],
                        ),
                    )
                else:
                    self.logger.info(f'[GEMINI] {self.model_name} --- THINK[{self.THINK_BUDGET}] --- TOOLS[NONE]')
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_text.text,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=self.THINK_BUDGET),
                            response_modalities=["TEXT"],
                        ),
                    )

            except Exception as e:
                print(f"Failed to init genai.Client for {self.model_name}: {e}")
                return "Failed to parse text"
        else:
            model = GenerativeModel(self.model_name)
            response = model.generate_content(prompt_text.text)
        self.log_response_metrics(response)
        return response.text
    
    def call_llm_api_GoogleGemini(self, prompt_template, json_report, paths):
        if paths is not None:
            _____, ____, _, __, ___, json_file_path_wiki, txt_file_path_ind_prompt = paths
        else:
            json_file_path_wiki = None
            txt_file_path_ind_prompt = None
            self.exit_early_for_JSON = True

        self.json_report = json_report
        if self.json_report:            
            self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        self.monitor.start_monitoring_usage()
        nt_in = 0
        nt_out = 0
        
        ind = 0
        while ind < self.MAX_RETRIES:
            ind += 1
            try:
                # model_kwargs = {"temperature": self.adjust_temp}
                # Invoke the chain to generate prompt text
                response = self.chain.invoke({"query": prompt_template})#, "model_kwargs": model_kwargs})

                # Use retry_parser to parse the response with retry logic
                output = self.retry_parser.parse_with_prompt(response, prompt_value=prompt_template)

                if output is None:
                    self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{response}')
                    self._adjust_config()
                else:
                    nt_in = count_tokens(prompt_template, self.VENDOR, self.TOKENIZER_NAME)
                    nt_out = count_tokens(response, self.VENDOR, self.TOKENIZER_NAME)

                    # self.logger.info(f"self.JSON_dict_structure\n{self.JSON_dict_structure}") #####################################################################################
                    # self.logger.info(f"############") #####################################################################################
                    # self.logger.info(f"output before\n{output}") #####################################################################################
                    output = validate_and_align_JSON_keys_with_template(output, self.JSON_dict_structure)
                    # self.logger.info(f"############") #####################################################################################
                    # self.logger.info(f"output after\n{output}") #####################################################################################

                    ### This allows VVGO to just get the JSON and exit
                    if self.exit_early_for_JSON and not self.exit_early_with_WFO:
                        return output, nt_in, nt_out, None, None, None, ""
                    
                    elif self.exit_early_for_JSON and self.exit_early_with_WFO:
                        _, WFO_record, __, ___ = run_tools(output, self.tool_WFO, self.tool_GEO, self.tool_wikipedia, json_file_path_wiki)
                        self.logger.info(f"WFO Record:\n{WFO_record}")
                        return output, nt_in, nt_out, None, None, None, WFO_record
                    
                    

                    if output is None:
                        self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{response}')
                        self._adjust_config() 
                    else:
                        self.monitor.stop_inference_timer() # Starts tool timer too

                        if self.json_report:            
                            self.json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                        
                        #############################################################   
                        # Temp for Shelly -- only run tools if has chromosome
                        # has_chromosome = output.get('chromosomeCount', '').strip()
                        # has_guardcell = output.get('guardCell', '').strip()
                        # if (not has_chromosome) and (not has_guardcell):
                        #     self.tool_WFO = False
                        #     self.tool_GEO = False
                        #     self.tool_wikipedia = False
                        # else:
                        #     self.tool_WFO = True
                        #     self.tool_GEO = True
                        #     self.tool_wikipedia = True
                        #############################################################    
                        output_WFO, WFO_record, output_GEO, GEO_record = run_tools(output, self.tool_WFO, self.tool_GEO, self.tool_wikipedia, json_file_path_wiki)

                        save_individual_prompt(sanitize_prompt(prompt_template), txt_file_path_ind_prompt)

                        self.logger.info(f"Formatted JSON:\n{json.dumps(output,indent=4)}")
                        
                        usage_report = self.monitor.stop_monitoring_report_usage()    

                        if self.adjust_temp != self.starting_temp:            
                            self._reset_config()

                        if self.json_report:            
                            self.json_report.set_text(text_main=f'LLM call successful')
                        return output, nt_in, nt_out, WFO_record, GEO_record, usage_report

            except Exception as e:
                self.logger.error(f'{e}')
                
                self._adjust_config()           
                time.sleep(self.RETRY_DELAY)

        self.logger.info(f"Failed to extract valid JSON after [{ind}] attempts")
        if self.json_report:            
            self.json_report.set_text(text_main=f'Failed to extract valid JSON after [{ind}] attempts')
        
        self.monitor.stop_inference_timer() # Starts tool timer too

        usage_report = self.monitor.stop_monitoring_report_usage()                
        self._reset_config()

        if self.json_report:            
            self.json_report.set_text(text_main=f'LLM call failed')
        return None, nt_in, nt_out, None, None, usage_report


