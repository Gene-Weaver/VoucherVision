import os, time, random, torch, json
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template


class MistralHandler: 
    RETRY_DELAY = 2  # Wait 10 seconds before retrying
    MAX_RETRIES = 5  # Maximum number of retries
    STARTING_TEMP = 0.5 #0.01
    TOKENIZER_NAME = None
    VENDOR = 'mistral'
    RANDOM_SEED = 2023

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation):
        self.cfg = cfg
        self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
        self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
        self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']

        self.logger = logger
        self.monitor = SystemLoadMonitor(logger)
        self.has_GPU = torch.cuda.is_available()        
        self.model_name = model_name
        self.JSON_dict_structure = JSON_dict_structure

        self.config_vals_for_permutation = config_vals_for_permutation
        
        # Set up a parser
        self.parser = JsonOutputParser()

        # Define the prompt template
        self.prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        self._set_config()

    def _set_config(self):
        if self.config_vals_for_permutation:
            self.starting_temp = float(self.config_vals_for_permutation.get('mistral').get('temperature'))
            self.config = {
                    'max_tokens': self.config_vals_for_permutation.get('mistral').get('max_tokens'),
                    'temperature': self.starting_temp,
                    'top_p': self.config_vals_for_permutation.get('mistral').get('top_p'),
                    'top_k': self.config_vals_for_permutation.get('mistral').get('top_k'),
                    'safe_mode': self.config_vals_for_permutation.get('mistral').get('safe_mode'),
                    'random_seed': self.config_vals_for_permutation.get('mistral').get('random_seed'),
                    }
        else:
            self.starting_temp = float(self.STARTING_TEMP)
            self.config = {
                'max_tokens': 1024,
                'temperature': self.starting_temp,
                'random_seed': self.RANDOM_SEED,
                'safe_mode': False,
                'top_p': 0.5,
                'top_k': 0.5,
            }

        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp 
        
        self._build_model_chain_parser()


    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        self.config['random_seed'] = random.randint(1, 1000) 
        if self.json_report:
            self.json_report.set_text(text_main=f'Incrementing temperature from {self.adjust_temp} to {new_temp} and random_seed to {self.config.get("random_seed")}')
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp} and random_seed to {self.config.get("random_seed")}')
        self.adjust_temp += self.temp_increment
        self.config['temperature'] = self.adjust_temp    

    def _reset_config(self):
        if self.json_report:
            self.json_report.set_text(text_main=f'Resetting temperature from {self.adjust_temp} to {self.starting_temp} and random_seed to {self.RANDOM_SEED}')
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {self.starting_temp} and random_seed to {self.RANDOM_SEED}')
        self.adjust_temp = self.starting_temp
        self.config['temperature'] = self.starting_temp    
        self.config['random_seed'] = self.RANDOM_SEED
          
    def _build_model_chain_parser(self):
        # Initialize MistralAI
        self.llm_model = ChatMistralAI(mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
                            model=self.model_name,
                            max_tokens=self.config.get('max_tokens'), 
                            safe_mode=self.config.get('safe_mode'), 
                            top_p=self.config.get('top_p'),
                            top_k=self.config.get('top_k'),
                            )
        
        # Set up the retry parser with the runnable
        self.retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.parser, llm=self.llm_model, max_retries=self.MAX_RETRIES)
        
        self.chain = self.prompt | self.llm_model    

    def call_llm_api_MistralAI(self, prompt_template, json_report, paths):
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
                # model_kwargs = {"temperature": self.adjust_temp, "random_seed": self.config.get("random_seed")}
                
                # Invoke the chain to generate prompt text
                response = self.chain.invoke({"query": prompt_template})#, "model_kwargs": model_kwargs})

                # Use retry_parser to parse the response with retry logic
                output = self.retry_parser.parse_with_prompt(response.content, prompt_value=prompt_template)

                if output is None:
                    self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{response}')
                    self._adjust_config()
                else:
                    nt_in = count_tokens(prompt_template, self.VENDOR, self.TOKENIZER_NAME)
                    nt_out = count_tokens(response.content, self.VENDOR, self.TOKENIZER_NAME)
                    
                    output = validate_and_align_JSON_keys_with_template(output, self.JSON_dict_structure)
                    if output is None:
                        self.logger.error(f'[Attempt {ind}] Failed to extract JSON from:\n{response}')
                        self._adjust_config()           
                    else:
                        self.monitor.stop_inference_timer() # Starts tool timer too
                        
                        if self.json_report:
                            self.json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
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
                self.logger.error(f'JSON Parsing Error (LangChain): {e}')
                
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
