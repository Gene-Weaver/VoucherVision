import os, time, random, torch, json, ssl
from typing import Any, Dict, List, Optional, cast
from langchain_classic.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, AzureMLEndpointApiType, LlamaChatContentFormatter
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_community.llms.azureml_endpoint import (
    AzureMLBaseEndpoint,
    AzureMLEndpointApiType,
    ContentFormatterBase,
)

from utils_LLM import run_tools, count_tokens, save_individual_prompt, sanitize_prompt
# from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
# from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template
from utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template


# https://api.python.langchain.com/en/latest/_modules/langchain_community/chat_models/azureml_endpoint.html#LlamaChatContentFormatter


class MistralChatContentFormatter(ContentFormatterBase):
    """Content formatter for `Mistral Models`."""

    SUPPORTED_ROLES: List[str] = ["user", "assistant"]

    @staticmethod
    def _convert_message_to_dict(message: BaseMessage) -> Dict:
        """Converts a message to a dict according to a role."""
        content = cast(str, message.content)
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

        print({
            "role": role,
            "content": ContentFormatterBase.escape_special_characters(content),
        })
        return {
            "role": role,
            "content": ContentFormatterBase.escape_special_characters(content),
        }

    @property
    def supported_api_types(self) -> List[AzureMLEndpointApiType]:
        return [AzureMLEndpointApiType.realtime, AzureMLEndpointApiType.serverless]

    def format_messages_request_payload(
        self,
        messages: List[BaseMessage],
        model_kwargs: Dict,
        api_type: AzureMLEndpointApiType,
    ) -> bytes:
        """Formats the request according to the chosen API."""
        input_string = [
            MistralChatContentFormatter._convert_message_to_dict(message)
            for message in messages
        ]
        request_payload = json.dumps(
            {
                "input_data": {
                    "input_string": input_string,
                    "parameters": model_kwargs,
                }
            }
        )
        return str.encode(request_payload)

    def format_response_payload(
        self,
        output: bytes,
        api_type: AzureMLEndpointApiType = AzureMLEndpointApiType.realtime,
    ) -> ChatGeneration:
        """Formats response according to the API type."""
        if api_type == AzureMLEndpointApiType.realtime:
            try:
                # choice = json.loads(output).get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                choice = json.loads(output).get("output", "").strip()
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError("Error processing the output.") from e
            return ChatGeneration(
                message=BaseMessage(
                    content=choice,
                    type="assistant",  # Assuming the response is always from the assistant
                ),
                generation_info=None,
            )
        else:
            raise ValueError(f"`api_type` {api_type} is not supported by this formatter")



class AzureMLChatOnlineEndpoint(BaseChatModel, AzureMLBaseEndpoint):
    """Azure ML Online Endpoint chat models.

    Example:
        .. code-block:: python
            azure_llm = AzureMLOnlineEndpoint(
                endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/score",
                endpoint_api_type=AzureMLApiType.realtime,
                endpoint_api_key="my-api-key",
                content_formatter=chat_content_formatter,
            )
    """  # noqa: E501

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azureml_chat_endpoint"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to an AzureML Managed Online endpoint.
        Args:
            messages: The messages in the conversation with the chat model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = azureml_model("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs.update(kwargs)
        if stop:
            _model_kwargs["stop"] = stop

        request_payload = self.content_formatter.format_messages_request_payload(
            messages, _model_kwargs, self.endpoint_api_type
        )
        response_payload = self.http_client.call(
            body=request_payload, run_manager=run_manager
        )
        generations = self.content_formatter.format_response_payload(
            response_payload, self.endpoint_api_type
        )
        return ChatResult(generations=[generations])


    def format_response_payload(self, output: bytes, output2) -> str:
        print(output)
        print(output2)
        # response_json = json.loads(output)
        return output#[0]["summary_text"]
    
class MistralHandlerAzureEndpoints: 
    RETRY_DELAY = 2  # Wait 10 seconds before retrying
    MAX_RETRIES = 5  # Maximum number of retries
    STARTING_TEMP = 0.1
    TOKENIZER_NAME = None
    VENDOR = 'mistral'
    RANDOM_SEED = 2023

    def __init__(self, model_name, JSON_dict_structure):
    # def __init__(self, cfg, logger, model_name, JSON_dict_structure):
        # self.cfg = cfg
        # self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
        # self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
        # self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']
        self.tool_WFO = True
        self.tool_GEO = True
        self.tool_wikipedia = True

        # Azure ML endpoint configuration
        self.endpoint_url = 'https://vouchervision-eqdxa.eastus2.inference.ml.azure.com/score' # 'https://vouchervision-mistral7b.eastus2.inference.ml.azure.com/score'# cfg['azure_ml']['endpoint_url']
        self.endpoint_api_key = 'Pdin5xjQJV5D85PKdWsmga4KMRFxuwjV' #'8VegSUnAoaFcST3GblrXflrVugDveWgh' # cfg['azure_ml']['api_key']
        self.endpoint_api_type = AzureMLEndpointApiType.realtime

        self.allow_self_signed_https(True)  # Call the method to allow self-signed certificates if necessary

        # self.logger = logger
        # self.monitor = SystemLoadMonitor(logger)
        self.has_GPU = torch.cuda.is_available()        
        self.model_name = model_name
        self.JSON_dict_structure = JSON_dict_structure
        self.starting_temp = float(self.STARTING_TEMP)
        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp 

        # Set up a parser
        self.parser = JsonOutputParser()

        # Define the prompt template
        self.prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        self._set_config()

    def allow_self_signed_https(self, allowed):
        """Bypass the server certificate verification on client side."""
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context


    def _set_config(self):
        self.config = {'max_tokens': 8024,
                'temperature': self.starting_temp,
                'random_seed': self.RANDOM_SEED,
                'safe_mode': False,
                'top_p': 1,
                'max_new_tokens':1024,
                }
        self._build_model_chain_parser()


    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        self.config['random_seed'] = random.randint(1, 1000) 
        # self.json_report.set_text(text_main=f'Incrementing temperature from {self.adjust_temp} to {new_temp} and random_seed to {self.config.get("random_seed")}')
        # self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp} and random_seed to {self.config.get("random_seed")}')
        self.adjust_temp += self.temp_increment
        self.config['temperature'] = self.adjust_temp    
        self._build_model_chain_parser()

    def _reset_config(self):
        # self.json_report.set_text(text_main=f'Resetting temperature from {self.adjust_temp} to {self.starting_temp} and random_seed to {self.RANDOM_SEED}')
        # self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {self.starting_temp} and random_seed to {self.RANDOM_SEED}')
        self.adjust_temp = self.starting_temp
        self.config['temperature'] = self.starting_temp    
        self.config['random_seed'] = self.RANDOM_SEED
          
    def _build_model_chain_parser(self):
        # Initialize MistralAI
        self.llm_model = AzureMLChatOnlineEndpoint(
            endpoint_url=self.endpoint_url,
            endpoint_api_type=self.endpoint_api_type,
            endpoint_api_key=self.endpoint_api_key,
            content_formatter=MistralChatContentFormatter() ,# LlamaChatContentFormatter(),
            # model_kwargs = {"temperature": self.adjust_temp, "random_seed": self.config.get("random_seed")}
            model_kwargs = {"temperature": self.adjust_temp, "max_new_tokens": 1024}
        )

        
        # Set up the retry parser with the runnable
        self.retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.parser, llm=self.llm_model, max_retries=self.MAX_RETRIES)
        
        self.chain = self.prompt | self.llm_model    

    def call_llm_api_MistralAI(self, prompt_template):#, json_report, paths):
        # _____, ____, _, __, ___, json_file_path_wiki, txt_file_path_ind_prompt = paths

        # self.json_report = json_report
        # self.json_report.set_text(text_main=f'Sending request to {self.model_name}')
        # self.monitor.start_monitoring_usage()
        nt_in = 0
        nt_out = 0

        ind = 0
        while ind < self.MAX_RETRIES:
            ind += 1
            try:
                # # Prepare the prompt as a HumanMessage
                # human_message = HumanMessage(content=prompt_template)
                
                # # Invoke Azure ML endpoint
                # response = self.chain.invoke([human_message])

                # output = self.parser.parse(response)  # Assuming your JsonOutputParser can handle the response format directly

                
                # Invoke the chain to generate prompt text
                # response = self.llm_model.invoke(prompt_template)
                response = self.chain.invoke({"query": prompt_template})

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

                        json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                        output_WFO, WFO_record, output_GEO, GEO_record = run_tools(output, self.tool_WFO, self.tool_GEO, self.tool_wikipedia, json_file_path_wiki)

                        save_individual_prompt(sanitize_prompt(prompt_template), txt_file_path_ind_prompt)

                        self.logger.info(f"Formatted JSON Pre-Sanitize:\n{json.dumps(output,indent=4)}")
                        
                        usage_report = self.monitor.stop_monitoring_report_usage()    

                        if self.adjust_temp != self.starting_temp:            
                            self._reset_config()

                        json_report.set_text(text_main=f'LLM call successful')
                        return output, nt_in, nt_out, WFO_record, GEO_record, usage_report

            except Exception as e:
                print(e)
                self.logger.error(f'JSON Parsing Error (LangChain): {e}')
                
                self._adjust_config()           
                time.sleep(self.RETRY_DELAY)

        self.logger.info(f"Failed to extract valid JSON after [{ind}] attempts")
        self.json_report.set_text(text_main=f'Failed to extract valid JSON after [{ind}] attempts')

        self.monitor.stop_inference_timer() # Starts tool timer too
        usage_report = self.monitor.stop_monitoring_report_usage()                
        self._reset_config()
        json_report.set_text(text_main=f'LLM call failed')

        return None, nt_in, nt_out, None, None, usage_report

if __name__ == '__main__':
    prompt_test = """Please help me complete this text parsing task given the following rules and unstructured OCR text. Your task is to refactor the OCR text into a structured JSON dictionary that matches the structure specified in the following rules. Please follow the rules strictly.
                The rules are:
                1. Refactor the unstructured OCR text into a dictionary based on the JSON structure outlined below. 2. Map the unstructured OCR text to the appropriate JSON key and populate the field given the user-defined rules. 3. JSON key values are permitted to remain empty strings if the corresponding information is not found in the unstructured OCR text. 4. 
Duplicate dictionary fields are not allowed. 5. Ensure all JSON keys are in camel case. 6. Ensure new JSON field values follow sentence case capitalization. 7. Ensure all key-value pairs in the JSON dictionary strictly adhere to the format and data types specified in the template. 8. Ensure output JSON string is valid JSON format. It should not have trailing commas or unquoted keys. 9. Only return a JSON dictionary represented as a string. You should not explain your answer.
                This section provides rules for formatting each JSON value organized by the JSON key.
                This is the JSON template that includes instructions for each key:
                {"catalogNumber": "Barcode identifier, typically a number with at least 6 digits, but fewer than 30 digits.", "order": "The full scientific name of the order in which 
the taxon is classified. Order must be capitalized.", "family": "The full scientific name of the family in which the taxon is classified. Family must be capitalized.", "scientificName": "The scientific name of the taxon including genus, specific epithet, and any lower classifications.", "scientificNameAuthorship": "The authorship information for the scientificName formatted according to the conventions of the applicable Darwin Core nomenclaturalCode.", "genus": "Taxonomic determination to genus. Genus must be capitalized. If genus is not present use the taxonomic family name followed by the word 'indet'.", "subgenus": "The full scientific name of the subgenus in which the taxon is classified. Values should include the genus to avoid homonym confusion.", "specificEpithet": "The name of the first or species epithet of the scientificName. Only include the species epithet.", "infraspecificEpithet": "The name of the lowest or terminal infraspecific epithet of the scientificName, excluding any rank designation.", "identifiedBy": "A comma separated list of names of people, groups, or organizations who assigned the taxon to the subject organism. This is not the specimen collector.", "recordedBy": "A comma separated list of names of people, groups, or organizations responsible for observing, recording, collecting, or presenting the original specimen. The primary collector or observer should be listed first.", "recordNumber": "An identifier given to 
the occurrence at the time it was recorded. Often serves as a link between field notes and an occurrence record, such as a specimen collector's number.", "verbatimEventDate": "The verbatim original representation of the date and time information for when the specimen was collected. Date of collection exactly as it appears on the label. Do not change the format or 
correct typos.", "eventDate": "Date the specimen was collected formatted as year-month-day, YYYY-MM_DD. If specific components of the date are unknown, they should be replaced with zeros. Examples \"0000-00-00\" if the entire date is unknown, \"YYYY-00-00\" if only the year is known, and \"YYYY-MM-00\" if year and month are known but day is not.", "habitat": "A category or description of the habitat in which the specimen collection event occurred.", "occurrenceRemarks": "Text describing the specimen's geographic location. Text describing the appearance of the specimen. A statement about the presence or absence of a taxon at a the collection location. Text describing the significance of the specimen, such as a specific expedition or notable collection. Description of plant features such as leaf shape, size, color, stem texture, height, flower structure, scent, fruit or seed characteristics, root system 
type, overall growth habit and form, any notable aroma or secretions, presence of hairs or bristles, and any other distinguishing morphological or physiological characteristics.", "country": "The name of the country or major administrative unit in which the specimen was originally collected.", "stateProvince": "The name of the next smaller administrative region than country (state, province, canton, department, region, etc.) in which the specimen was originally collected.", "county": "The full, unabbreviated name of the next smaller administrative region than stateProvince (county, shire, department, parish etc.) in which the specimen was originally collected.", "municipality": "The full, unabbreviated name of the next smaller administrative region than county (city, municipality, etc.) in which the specimen was originally collected.", "locality": "Description of geographic location, landscape, landmarks, regional features, nearby places, or any contextual information aiding in pinpointing the exact origin or location of the specimen.", "degreeOfEstablishment": "Cultivated plants are intentionally grown by humans. In text descriptions, look for planting dates, garden locations, ornamental, cultivar names, garden, or farm to indicate cultivated plant. Use either - unknown or cultivated.", "decimalLatitude": "Latitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format.", "decimalLongitude": "Longitude decimal coordinate. Correct and convert the verbatim location coordinates to conform with the decimal degrees GPS coordinate format.", "verbatimCoordinates": "Verbatim location coordinates as they appear on the label. Do not convert formats. Possible coordinate types include [Lat, Long, UTM, TRS].", "minimumElevationInMeters": "Minimum elevation or altitude in meters. Only if units are explicit then convert from feet (\"ft\" or \"ft.\"\" or \"feet\") to meters (\"m\" or \"m.\" or \"meters\"). Round to integer.", "maximumElevationInMeters": "Maximum elevation or altitude in meters. If only one elevation is present, then max_elevation should be set to the null_value. Only if units are 
explicit then convert from feet (\"ft\" or \"ft.\" or \"feet\") to meters (\"m\" or \"m.\" or \"meters\"). Round to integer."}
                The unstructured OCR text is:
                OCR

Google Handwritten OCR
RANCHO SANTA ANA BOTANIC GARDEN
DistichtiLANTEN MET (L.) Greene
OF MEXICO
Nayarit Isabel Island, E of Tres Marias Is.
Volcanic islet--cormorant and tern breeding
ground.
THE
In open, level to slightly sloping areas
favored by the terns for nesting, hear the
beach; surrounded by low forest consisting
almost entirely of Crataeva Tapia (3-4 m tall),
Porous soil.
MICH
UNIVERSITY OF MICHIC
1817
copyright reserved 122841 cm
29 April 1973
DEC 1 8 1975
University of Michigan Herbarium
1122841
                Please populate the following JSON dictionary based on the rules and the unformatted OCR text:
                {'catalogNumber': '', 'order': '', 'family': '', 'scientificName': '', 'scientificNameAuthorship': '', 'genus': '', 'subgenus': '', 'specificEpithet': '', 'infraspecificEpithet': '', 'identifiedBy': '', 'recordedBy': '', 'recordNumber': '', 'verbatimEventDate': '', 'eventDate': '', 'habitat': '', 'occurrenceRemarks': '', 'country': '', 'stateProvince': '', 'county': '', 'municipality': '', 'locality': '', 'degreeOfEstablishment': '', 'decimalLatitude': '', 'decimalLongitude': '', 'verbatimCoordinates': '', 'minimumElevationInMeters': '', 'maximumElevationInMeters': ''}"""
    JSON_dict_structure = {'catalogNumber': '', 'order': '', 'family': '', 'scientificName': '', 'scientificNameAuthorship': '', 'genus': '', 'subgenus': '', 'specificEpithet': '', 'infraspecificEpithet': '', 'identifiedBy': '', 'recordedBy': '', 'recordNumber': '', 'verbatimEventDate': '', 'eventDate': '', 'habitat': '', 'occurrenceRemarks': '', 'country': '', 'stateProvince': '', 'county': '', 'municipality': '', 'locality': '', 'degreeOfEstablishment': '', 'decimalLatitude': '', 'decimalLongitude': '', 'verbatimCoordinates': '', 'minimumElevationInMeters': '', 'maximumElevationInMeters': ''}
    ml = MistralHandlerAzureEndpoints('mistral', JSON_dict_structure)
    ml.call_llm_api_MistralAI(prompt_test, )