import openai
import os, json, sys, inspect, tiktoken, time, requests
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import HumanMessage
from general_utils import get_cfg_from_full_path

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from prompts import PROMPT_UMICH_skeleton_all_asia, PROMPT_OCR_Organized, PROMPT_UMICH_skeleton_all_asia_GPT4, PROMPT_OCR_Organized_GPT4, PROMPT_JSON
from prompt_catalog import PromptCatalog

RETRY_DELAY = 61  # Wait 60 seconds before retrying
MAX_RETRIES = 5  # Maximum number of retries


def azure_call(model, messages):
    response = model(messages=messages)
    return response

def OCR_to_dict(is_azure, logger, MODEL, prompt, llm, prompt_version):
    for i in range(MAX_RETRIES):
        try:
            do_use_SOP = True

            if do_use_SOP:
                logger.info(f'Waiting for {MODEL} API call --- Using StructuredOutputParser')
                response = structured_output_parser(is_azure, MODEL, llm, prompt, logger, prompt_version)
                if response is None:
                    return None
                else:
                    return response['Dictionary']

            else:
                ### Direct GPT ###
                logger.info(f'Waiting for {MODEL} API call')
                if not is_azure:
                    response = openai.ChatCompletion.create(
                        model=MODEL,
                        temperature = 0,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant acting as a transcription expert and your job is to transcribe herbarium specimen labels based on OCR data and reformat it to meet Darwin Core Archive Standards into a Python dictionary based on certain rules."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=4096,
                    )
                    # print the model's response
                    return response.choices[0].message['content']
                else:
                    msg = HumanMessage(
                        content=prompt
                    )
                    response = azure_call(llm, [msg])
                    return response.content
        except Exception as e:
            logger.error(f'{e}')
            if i < MAX_RETRIES - 1:  # No delay needed after the last try
                time.sleep(RETRY_DELAY)
            else:
                raise

# def OCR_to_dict(logger, MODEL, prompt, OCR, BASE_URL, HEADERS):
#     for i in range(MAX_RETRIES):
#         try:
#             do_use_SOP = False

#             if do_use_SOP:
#                 logger.info(f'Waiting for {MODEL} API call --- Using StructuredOutputParser -- Content')
#                 response = structured_output_parser(MODEL, OCR, prompt, logger)
#                 if response is None:
#                     return None
#                 else:
#                     return response['Dictionary']

#             else:
#                 ### Direct GPT through Azure ###
#                 logger.info(f'Waiting for {MODEL} API call')
#                 response = azure_gpt_request(prompt, BASE_URL, HEADERS, model_name=MODEL)

#                 # Handle the response data. Note: You might need to adjust the following line based on the exact response format of the Azure API.
#                 content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
#                 return content
#         except requests.exceptions.RequestException as e:  # Replace openai.error.APIError with requests exception.
#             # Handle HTTP exceptions. You can adjust this based on the Azure API's error responses.
#             if e.response.status_code == 502:
#                 logger.info(f'   ***    502 error was encountered, wait and try again   ***')
#                 if i < MAX_RETRIES - 1:
#                     time.sleep(RETRY_DELAY)
#             else:
#                 raise


def OCR_to_dict_16k(is_azure, logger, MODEL, prompt, llm, prompt_version):
    for i in range(MAX_RETRIES):
        try:
            fs = FunctionSchema()
            response = openai.ChatCompletion.create(
                model=MODEL,
                temperature = 0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant acting as a transcription expert and your job is to transcribe herbarium specimen labels based on OCR data and reformat it to meet Darwin Core Archive Standards into a Python dictionary based on certain rules."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8000,
                function_call= "none",
                functions= fs.format_C21_AA_V1()

            )
            # Try to parse the response into JSON
            call_failed = False
            try:
                response_string = response.choices[0].message['content']
            except:
                call_failed = True
                response_string = prompt

            if not call_failed:
                try:
                    # Try to parse the response into JSON
                    response_dict = json.loads(response_string)
                    return response_dict['Dictionary']
                except json.JSONDecodeError:
                    # If the response is not a valid JSON, call the structured_output_parser_for_function_calls_fail function
                    logger.info(f'Invalid JSON response, calling structured_output_parser_for_function_calls_fail function')
                    logger.info(f'Waiting for {MODEL} API call --- Using StructuredOutputParser --- JSON Fixer')
                    response_sop = structured_output_parser_for_function_calls_fail(is_azure, MODEL, response_string, logger, llm, prompt_version, is_helper=False)
                    if response_sop is None:
                        return None
                    else:
                        return response_sop['Dictionary']
            else:
                try:
                    logger.info(f'Call Failed. Attempting fallback JSON parse without guidance')
                    logger.info(f'Waiting for {MODEL} API call --- Using StructuredOutputParser --- JSON Fixer')
                    response_sop = structured_output_parser_for_function_calls_fail(is_azure, MODEL, response_string, logger, llm, prompt_version, is_helper=False)
                    if response_sop is None:
                        return None
                    else:
                        return response_sop['Dictionary']
                except:
                    return None
        except Exception as e:
            # if e.status_code == 401: # or you can check the error message
            logger.info(f'   ***    401 error was encountered, wait and try again   ***')
            # If a 401 error was encountered, wait and try again
            if i < MAX_RETRIES - 1:  # No delay needed after the last try
                time.sleep(RETRY_DELAY)
            else:
                # If it was a different error, re-raise it
                raise
            
def structured_output_parser(is_azure, MODEL, llm, prompt_template, logger, prompt_version, is_helper=False):
    if not is_helper:
        response_schemas = [
            ResponseSchema(name="SpeciesName", description="Taxonomic determination, genus_species"),
            ResponseSchema(name="Dictionary", description='Formatted JSON object'),]#prompt_template),]
    elif is_helper:
        response_schemas = [
            ResponseSchema(name="Dictionary", description='Formatted JSON object'),#prompt_template),
            ResponseSchema(name="Summary", description="A one sentence summary of the content"),]
        
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("Parse the OCR text into the correct structured format.\n{format_instructions}\n{question}")  
        ],
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    # Handle Azure vs OpenAI implementation
    if is_azure:
        _input = prompt.format_prompt(question=prompt_template)
        msg = HumanMessage(content=_input.to_string())
        output = azure_call(llm, [msg])
    else:
        chat_model = ChatOpenAI(temperature=0, model=MODEL)
        _input = prompt.format_prompt(question=prompt_template)
        output = chat_model(_input.to_messages())

    # Log token length if running with Gradio
    try:
        nt = num_tokens_from_string(_input.to_string(), "cl100k_base")
        logger.info(f'Prompt token length --- {nt}')
    except:
        pass

    # Parse the output
    try:
        # Check if output is of type 'ai' and parse accordingly
        if output.type == 'ai':
            parsed_content = output.content
            logger.info(f'Formatted JSON\n{parsed_content}')
        else:
            # If not 'ai', log and set parsed_content to None or a default value
            logger.error('Output type is not "ai". Unable to parse.')
            return None

        # Clean up the parsed content
        parsed_content = parsed_content.replace('\n', "").replace('\t', "").replace('|', "")

        # Attempt to parse the cleaned content
        try:
            refined_response = output_parser.parse(parsed_content)
            return refined_response
        except Exception as parse_error:
            # Handle parsing errors specifically
            logger.error(f'Parsing Error: {parse_error}')
            return structured_output_parser_for_function_calls_fail(is_azure, MODEL, parsed_content, logger, llm, prompt_version, is_helper)

    except Exception as e:
        # Handle any other exceptions that might occur
        logger.error(f'Unexpected Error: {e}')
        return None

def structured_output_parser_for_function_calls_fail(is_azure, MODEL, failed_response, logger, llm, prompt_version, is_helper=False, try_ind=0):
    if try_ind > 5:
        return None

    # prompt_redo = PROMPT_JSON('helper' if is_helper else 'dict', failed_response)
    Prompt = PromptCatalog()
    if prompt_version in ['prompt_v1_verbose', 'prompt_v1_verbose_noDomainKnowledge']:
        prompt_redo = Prompt.prompt_gpt_redo_v1(failed_response)
    elif prompt_version in ['prompt_v2_json_rules']:
        prompt_redo = Prompt.prompt_gpt_redo_v2(failed_response)
    else: raise

    response_schemas = [
        ResponseSchema(name="Summary", description="A one sentence summary of the content"),
        ResponseSchema(name="Dictionary", description='Formatted JSON object')
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("The following text contains JSON formatted text, but there is an error that you need to correct.\n{format_instructions}\n{question}")
        ],
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    _input = prompt.format_prompt(question=prompt_redo)

    # Log token length if running with Gradio
    try:
        nt = num_tokens_from_string(_input.to_string(), "cl100k_base")
        logger.info(f'Prompt Redo token length --- {nt}')
    except:
        pass

    if is_azure:
        msg = HumanMessage(content=_input.to_string())
        output = azure_call(llm, [msg])
    else:
        chat_model = ChatOpenAI(temperature=0, model=MODEL)
        output = chat_model(_input.to_messages())

    try:
        refined_response = output_parser.parse(output.content)
    except json.decoder.JSONDecodeError as e:
        try_ind += 1
        error_message = str(e)
        redo_content = f'The error messsage is: {error_message}\nThe broken JSON object is: {output.content}'
        logger.info(f'[Failed JSON Object]\n{output.content}')
        refined_response = structured_output_parser_for_function_calls_fail(is_azure, MODEL, redo_content, logger, llm, prompt_version, is_helper, try_ind)
    except:
        try_ind += 1
        logger.info(f'[Failed JSON Object]\n{output.content}')
        refined_response = structured_output_parser_for_function_calls_fail(is_azure, MODEL, output.content, logger, llm, prompt_version, is_helper, try_ind)

    return refined_response



def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens





class FunctionSchema:
    def __init__(self):
        pass

    def format_C21_AA_V1(self):
        return [
            {
                "name": "format_C21_AA_V1",
                "description": "Format the given data into a specific dictionary",
                "parameters": {
                    "type": "object",
                    "properties": {},  # specify parameters here if your function requires any
                    "required": []  # list of required parameters
                },
                "output_type": "json",
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "Dictionary": {
                            "type": "object",
                            "properties": {
                                "Catalog Number": {"type": "array", "items": {"type": "string"}},
                                "Genus": {"type": "array", "items": {"type": "string"}},
                                "Species": {"type": "array", "items": {"type": "string"}},
                                "subspecies": {"type": "array", "items": {"type": "string"}},
                                "variety": {"type": "array", "items": {"type": "string"}},
                                "forma": {"type": "array", "items": {"type": "string"}},
                                "Country": {"type": "array", "items": {"type": "string"}},
                                "State": {"type": "array", "items": {"type": "string"}},
                                "County": {"type": "array", "items": {"type": "string"}},
                                "Locality Name": {"type": "array", "items": {"type": "string"}},
                                "Min Elevation": {"type": "array", "items": {"type": "string"}},
                                "Max Elevation": {"type": "array", "items": {"type": "string"}},
                                "Elevation Units": {"type": "array", "items": {"type": "string"}},
                                "Verbatim Coordinates": {"type": "array", "items": {"type": "string"}},
                                "Datum": {"type": "array", "items": {"type": "string"}},
                                "Cultivated": {"type": "array", "items": {"type": "string"}},
                                "Habitat": {"type": "array", "items": {"type": "string"}},
                                "Collectors": {"type": "array", "items": {"type": "string"}},
                                "Collector Number": {"type": "array", "items": {"type": "string"}},
                                "Verbatim Date": {"type": "array", "items": {"type": "string"}},
                                "Date": {"type": "array", "items": {"type": "string"}},
                                "End Date": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "SpeciesName": {
                            "type": "object",
                            "properties": {
                                "taxonomy": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                }
            }
        ]

    def format_C21_AA_V1_helper(self):
        return [
            {
                "name": "format_C21_AA_V1_helper",
                "description": "Helper function for format_C21_AA_V1 to further format the given data",
                "parameters": {
                    "type": "object",
                    "properties": {},  # specify parameters here if your function requires any
                    "required": []  # list of required parameters
                },
                "output_type": "json",
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "Dictionary": {
                            "type": "object",
                            "properties": {
                                "TAXONOMY": {
                                    "type": "object",
                                    "properties": {
                                        "Order": {"type": "array", "items": {"type": "string"}},
                                        "Family": {"type": "array", "items": {"type": "string"}},
                                        "Genus":{"type": "array", "items": {"type": "string"}},
                                        "Species": {"type": "array", "items": {"type": "string"}},
                                        "Subspecies": {"type": "array", "items": {"type": "string"}},
                                        "Variety": {"type": "array", "items": {"type": "string"}},
                                        "Forma": {"type": "array", "items": {"type": "string"}},
                                    }
                                },
                                "GEOGRAPHY": {
                                    "type": "object",
                                    "properties": {
                                        "Country": {"type": "array", "items": {"type": "string"}},
                                        "State": {"type": "array", "items": {"type": "string"}},
                                        "Prefecture": {"type": "array", "items": {"type": "string"}},
                                        "Province": {"type": "array", "items": {"type": "string"}},
                                        "District": {"type": "array", "items": {"type": "string"}},
                                        "County": {"type": "array", "items": {"type": "string"}},
                                        "City": {"type": "array", "items": {"type": "string"}},
                                        "Administrative Division": {"type": "array", "items": {"type": "string"}},
                                    }
                                },
                                "LOCALITY": {
                                    "type": "object",
                                    "properties": {
                                        "Landscape": {"type": "array", "items": {"type": "string"}},
                                        "Nearby Places": {"type": "array", "items": {"type": "string"}},
                                    }
                                },
                                "COLLECTING": {
                                    "type": "object",
                                    "properties": {
                                        "Collector": {"type": "array", "items": {"type": "string"}},
                                        "Collector's Number": {"type": "array", "items": {"type": "string"}},
                                        "Verbatim Date": {"type": "array", "items": {"type": "string"}},
                                        "Formatted Date": {"type": "array", "items": {"type": "string"}},
                                        "Cultivation Status": {"type": "array", "items": {"type": "string"}},
                                        "Habitat Description": {"type": "array", "items": {"type": "string"}},
                                    }
                                },
                                "MISCELLANEOUS": {
                                    "type": "object",
                                    "properties": {
                                        "Additional Information": {"type": "array", "items": {"type": "string"}},
                                    }
                                }
                            }
                        },
                        "Summary": {
                            "type": "object",
                            "properties": {
                                "Content Summary": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                }
            }
        ]
