import os, time, random, torch, requests, json
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template

class HyperbolicHandler:
    RETRY_DELAY = 2  # Wait 2 seconds before retrying
    MAX_RETRIES = 5  # Maximum number of retries
    STARTING_TEMP = 0.7
    RANDOM_SEED = 2023

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation):
        self.cfg = cfg
        self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
        self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
        self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']

        self.logger = logger
        self.monitor = SystemLoadMonitor(logger)
        self.JSON_dict_structure = JSON_dict_structure

        self.api_url = "https://api.hyperbolic.xyz/v1/completions"
        self.api_key = os.getenv("HYPERBOLIC_API_KEY")
        self.model_name = model_name
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.config_vals_for_permutation = config_vals_for_permutation
        self.parser = JsonOutputParser()
        self.prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self._set_config()

    def _set_config(self):
        """Set configuration values dynamically."""
        if self.config_vals_for_permutation:
            self.starting_temp = self.config_vals_for_permutation.get('hyperbolic', {}).get('temperature', self.STARTING_TEMP)
            self.config = {
                'max_tokens': self.config_vals_for_permutation.get('hyperbolic', {}).get('max_tokens', 1024),
                'temperature': self.starting_temp,
                'top_p': self.config_vals_for_permutation.get('hyperbolic', {}).get('top_p', 0.9),
                'random_seed': self.config_vals_for_permutation.get('hyperbolic', {}).get('random_seed', self.RANDOM_SEED),
            }
        else:
            self.starting_temp = self.STARTING_TEMP
            self.config = {
                'max_tokens': 1024,
                'temperature': self.starting_temp,
                'top_p': 0.9,
                'random_seed': self.RANDOM_SEED,
            }
        self.temp_increment = 0.2
        self.adjust_temp = self.starting_temp

    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        self.config['random_seed'] = random.randint(1, 1000)
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp} and random_seed to {self.config["random_seed"]}')
        self.adjust_temp += self.temp_increment
        self.config['temperature'] = self.adjust_temp
    
    def _reset_config(self):
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp} and random_seed to {self.RANDOM_SEED}')
        self.adjust_temp = self.starting_temp
        self.config['temperature'] = self.starting_temp
        self.config['random_seed'] = self.RANDOM_SEED

    def _send_request(self, prompt):
        """Send request to Hyperbolic API."""
        payload = {
            "prompt": prompt,
            "model": self.model_name,
            "max_tokens": self.config['max_tokens'],
            "temperature": self.config['temperature'],
            "top_p": self.config['top_p']
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            response_json = response.json()

            if "choices" in response_json:
                raw_response = response_json["choices"][0]["text"]
            else:
                raw_response = None

            if "usage" in response_json:
                usage = response_json["usage"]
                tokens_in = usage["prompt_tokens"]
                tokens_out = usage["completion_tokens"] 
            else:
                usage = None
                tokens_in = 0
                tokens_out = 0

            return self._validate_and_clean_response(raw_response), tokens_in, tokens_out
        else:
            response.raise_for_status()

    # def _build_model_chain_parser(self):
    #     """Set up the retry parser."""
    #     self.retry_parser = RetryWithErrorOutputParser.from_llm(
    #         parser=self.parser, llm=self._send_request, max_retries=self.MAX_RETRIES
    #     )
    #     self.chain = self.prompt | self.retry_parser
    
    def _validate_and_clean_response(self, raw_response):
        """Extract and validate JSON between the first '{' and the first '}'."""
        try:
            # Find the first opening brace and the first closing brace
            start_index = raw_response.find('{')
            end_index = raw_response.find('}', start_index)

            # Ensure both braces are found and the order is correct
            if start_index == -1 or end_index == -1 or start_index >= end_index:
                self.logger.error("No valid JSON found in the response.")
                return None

            # Extract the substring containing the potential JSON
            possible_json = raw_response[start_index:end_index + 1].strip()

            # Validate the JSON by loading it into a Python dictionary
            json_response = json.loads(possible_json)

            # Return the validated JSON as a string
            return json.dumps(json_response, indent=2)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during JSON cleaning/parsing: {e}")
            return None

        
    def call_llm_api_Hyperbolic(self, prompt_template, json_report, paths):
        _, _, _, _, _, json_file_path_wiki, txt_file_path_ind_prompt = paths

        self.json_report = json_report
        if self.json_report:
            self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        self.monitor.start_monitoring_usage()

        for attempt in range(self.MAX_RETRIES):
            try:
                # Build the full prompt
                # prompt = self.prompt.format(query=prompt_template)

                # Send the prompt and parse the response
                raw_response, tokens_in, tokens_out = self._send_request(prompt_template)
                if raw_response is None:
                    self.logger.error(f'[Attempt {attempt + 1}] Failed to extract valid JSON')
                    self._adjust_config()
                    continue

                parsed_response = self.parser.parse(raw_response)

                # Validate and align JSON
                output = validate_and_align_JSON_keys_with_template(parsed_response, self.JSON_dict_structure)
                
                if output is None:
                    self.logger.error(f'[Attempt {attempt + 1}] Failed to extract valid JSON')
                    self._adjust_config()
                    continue

                nt_in = tokens_in
                nt_out = tokens_out
                
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
                self.logger.error(f'JSON Parsing Error (LangChain): {e}')
                
                self._adjust_config()           
                time.sleep(self.RETRY_DELAY)

        self.logger.info(f"Failed to extract valid JSON after [{attempt}] attempts")
        if self.json_report:
            self.json_report.set_text(text_main=f'Failed to extract valid JSON after [{attempt}] attempts')

        self.monitor.stop_inference_timer() # Starts tool timer too
        usage_report = self.monitor.stop_monitoring_report_usage()                
        self._reset_config()
        if self.json_report:
            self.json_report.set_text(text_main=f'LLM call failed')

        return None, nt_in, nt_out, None, None, usage_report
