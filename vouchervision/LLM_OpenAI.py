import time, torch, json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import RetryWithErrorOutputParser

from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template

class OpenAIHandler: 
    RETRY_DELAY = 10  # Wait 10 seconds before retrying
    MAX_RETRIES = 3  # Maximum number of retries
    STARTING_TEMP = 0.5 # 0.5, config_vals_for_permutation
    TOKENIZER_NAME = 'gpt-4'
    VENDOR = 'openai'

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, is_azure, llm_object, config_vals_for_permutation):
        self.cfg = cfg
        self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
        self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
        self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']

        self.logger = logger
        self.model_name = model_name
        self.JSON_dict_structure = JSON_dict_structure
        self.is_azure = is_azure
        self.llm_object = llm_object
        self.name_parts = self.model_name.lower().split('-')
        
        self.monitor = SystemLoadMonitor(logger)
        self.has_GPU = torch.cuda.is_available() 

        ### Config
        self.config_vals_for_permutation = config_vals_for_permutation
        
        # Set up a parser
        self.parser = JsonOutputParser()

        self.prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self._set_config()

    def _can_use_json_mode(self):
        if self.is_azure:
            return False
        # gpt-4-turbo-preview (gpt-4-0125-preview)
        if ('0125' in self.name_parts) and ('4' in self.name_parts):
            return True
        # gpt-3.5-turbo-0125
        elif ('0125' in self.name_parts) and ('3.5' in self.name_parts) and ('turbo' in self.name_parts):
            return True
        else:
            return False


    def _set_config(self):
        if self.config_vals_for_permutation:
            self.starting_temp = float(self.config_vals_for_permutation.get('openai').get('temperature'))
            self.model_kwargs = {
                    'max_tokens': self.config_vals_for_permutation.get('openai').get('max_tokens'),
                    'temperature': self.starting_temp,
                    # 'seed': self.config_vals_for_permutation.get('openai').get('seed'),
                    'top_p': self.config_vals_for_permutation.get('openai').get('top_p'),
                    }
        else:
            self.starting_temp = float(self.STARTING_TEMP)
            self.model_kwargs = {
                    'max_tokens': 1024,
                    'temperature': self.starting_temp,
                    # 'seed': 2023,
                    'top_p': 1, # Set to 1, change temp only
                    }
        
        ### Not all openai models support json mode
        if self._can_use_json_mode():
            self.model_kwargs.update({"response_format": {"type": "json_object"}})
            
        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp 

        # Adjusting the LLM settings based on whether Azure is used
        if self.is_azure:
            self.llm_object.deployment_name = self.model_name
            self.llm_object.model_name = self.model_name
        else:
            self.llm_object = None
        self._build_model_chain_parser()


       # Define a function to format the input for azure_call
    def format_input_for_azure(self, prompt_text):
        msg = HumanMessage(content=prompt_text.text)
        # self.llm_object.temperature = self.config.get('temperature')
        return self.llm_object(messages=[msg]) 

    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        if self.json_report:            
            self.json_report.set_text(text_main=f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.adjust_temp += self.temp_increment
        self.model_kwargs['temperature'] = self.adjust_temp   

    def _reset_config(self):
        if self.json_report:            
            self.json_report.set_text(text_main=f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.adjust_temp = self.starting_temp
        self.model_kwargs['temperature'] = self.starting_temp   
        
    def _build_model_chain_parser(self):
        if not self.is_azure and ('instruct' in self.name_parts):
            # Determine the LLM to use based on whether this is an Azure instance
            if self.is_azure:
                llm_to_use = self.llm_object
            else:
                llm_to_use = OpenAI(
                    model=self.model_name,
                    temperature=self.model_kwargs.get('temperature'),
                    top_p=self.model_kwargs.get('top_p'),
                    max_tokens=self.model_kwargs.get('max_tokens')
                )
            # Set up the retry parser with 3 retries
            self.retry_parser = RetryWithErrorOutputParser.from_llm(
                parser=self.parser,
                llm=llm_to_use,
                max_retries=self.MAX_RETRIES
            )
        else:
            # Determine the LLM to use for non-Azure instances
            if self.is_azure:
                llm_to_use = self.llm_object
                self.llm_object.temperature = self.model_kwargs.get('temperature')
                self.llm_object.max_tokens = self.model_kwargs.get('max_tokens')
                self.llm_object.model_kwargs = self.model_kwargs 
            else:
                llm_to_use = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.model_kwargs.get('temperature'),
                    top_p=self.model_kwargs.get('top_p'),
                    max_tokens=self.model_kwargs.get('max_tokens'),
                )
            # Set up the retry parser with 3 retries for other cases
            self.retry_parser = RetryWithErrorOutputParser.from_llm(
                parser=self.parser,
                llm=llm_to_use,
                max_retries=self.MAX_RETRIES
            )
            
        # Prepare the chain
        if self.is_azure:
            chain_llm_to_use = self.format_input_for_azure
        else:
            if 'instruct' in self.name_parts:
                chain_llm_to_use = OpenAI(
                    model=self.model_name,
                    temperature=self.model_kwargs.get('temperature'),
                    top_p=self.model_kwargs.get('top_p'),
                    max_tokens=self.model_kwargs.get('max_tokens')
                )
            else:
                chain_llm_to_use = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.model_kwargs.get('temperature'),
                    top_p=self.model_kwargs.get('top_p'),
                    max_tokens=self.model_kwargs.get('max_tokens')
                )
        self.chain = self.prompt | chain_llm_to_use


    def call_llm_api_OpenAI(self, prompt_template, json_report, paths):
        _____, ____, _, __, ___, json_file_path_wiki, txt_file_path_ind_prompt = paths
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
                self.logger.info(str(self.model_kwargs))
                # Invoke the chain to generate prompt text
                response = self.chain.invoke(input={"query": prompt_template})#, **self.model_kwargs)# "model_kwargs": self.model_kwargs})

                response_text = response.content if not isinstance(response, str) else response

                # Use retry_parser to parse the response with retry logic
                try:
                    output = self.retry_parser.parse_with_prompt(response_text, prompt_value=prompt_template)
                except:
                    try:
                        output = json.loads(response_text)
                    except:
                        output = None

                if output is None:
                    self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{response_text}')
                    self._adjust_config()
                else:
                    nt_in = count_tokens(prompt_template, self.VENDOR, self.TOKENIZER_NAME)
                    nt_out = count_tokens(response_text, self.VENDOR, self.TOKENIZER_NAME)
                
                    output = validate_and_align_JSON_keys_with_template(output, self.JSON_dict_structure)
                    if output is None:
                        self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{response_text}')
                        self._adjust_config()   
                    else:
                        self.monitor.stop_inference_timer() # Starts tool timer too

                        if self.json_report:            
                            self.json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                        
                        output_WFO, WFO_record, output_GEO, GEO_record = run_tools(output, self.tool_WFO, self.tool_GEO, self.tool_wikipedia, json_file_path_wiki)

                        save_individual_prompt(sanitize_prompt(prompt_template), txt_file_path_ind_prompt)

                        self.logger.info(f"Formatted JSON Pre-Sanitize:\n{json.dumps(output,indent=4)}")

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


