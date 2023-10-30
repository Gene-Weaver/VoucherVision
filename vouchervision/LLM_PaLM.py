import os
import sys
import inspect
import json
from json import JSONDecodeError
import tiktoken
import random 
import google.generativeai as palm

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from prompts import PROMPT_PaLM_UMICH_skeleton_all_asia, PROMPT_PaLM_OCR_Organized, PROMPT_PaLM_Redo
from prompt_catalog import PromptCatalog
"""
Safety setting regularly block a response, so set to 4 to disable

class HarmBlockThreshold(Enum):
    HARM_BLOCK_THRESHOLD_UNSPECIFIED = 0
    BLOCK_LOW_AND_ABOVE = 1
    BLOCK_MEDIUM_AND_ABOVE = 2
    BLOCK_ONLY_HIGH = 3
    BLOCK_NONE = 4
"""

def OCR_to_dict_PaLM(logger, OCR, prompt_version, VVE):
    try:
        logger.info(f'Length of OCR raw -- {len(OCR)}')
    except:
        print(f'Length of OCR raw -- {len(OCR)}')
        
    # prompt = PROMPT_PaLM_UMICH_skeleton_all_asia(OCR, in_list, out_list) # must provide examples to PaLM differently than for chatGPT, at least 2 examples
    Prompt = PromptCatalog(OCR) 
    if prompt_version in ['prompt_v2_palm2']:
        version = 'v2'
        prompt = Prompt.prompt_v2_palm2(OCR)
    
    elif prompt_version in ['prompt_v1_palm2',]:
        version = 'v1'
        # create input: output: for PaLM
        # Find a similar example from the domain knowledge
        domain_knowledge_example = VVE.query_db(OCR, 4)
        similarity= VVE.get_similarity()
        domain_knowledge_example_string = json.dumps(domain_knowledge_example)
        in_list, out_list = create_OCR_analog_for_input(domain_knowledge_example)
        prompt = Prompt.prompt_v1_palm2(in_list, out_list, OCR)

    elif prompt_version in ['prompt_v1_palm2_noDomainKnowledge',]:
        version = 'v1'
        prompt = Prompt.prompt_v1_palm2_noDomainKnowledge(OCR)
    else:
        raise


    nt = num_tokens_from_string(prompt, "cl100k_base")
    # try:
    logger.info(f'Prompt token length --- {nt}')
    # except:
        # print(f'Prompt token length --- {nt}')

    do_use_SOP = False ########

    if do_use_SOP:
        '''TODO: Check back later to see if LangChain will support PaLM'''
        # logger.info(f'Waiting for PaLM API call --- Using StructuredOutputParser')
        # response = structured_output_parser(OCR, prompt, logger)
        # return response['Dictionary']
        pass

    else:
        # try:
        logger.info(f'Waiting for PaLM 2 API call')
        # except:
            # print(f'Waiting for PaLM 2 API call --- Content')

        safety_thresh = 4
        PaLM_settings = {'model': 'models/text-bison-001','temperature': 0,'candidate_count': 1,'top_k': 40,'top_p': 0.95,'max_output_tokens': 8000,'stop_sequences': [],
                         'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":safety_thresh},{"category":"HARM_CATEGORY_TOXICITY","threshold":safety_thresh},{"category":"HARM_CATEGORY_VIOLENCE","threshold":safety_thresh},{"category":"HARM_CATEGORY_SEXUAL","threshold":safety_thresh},{"category":"HARM_CATEGORY_MEDICAL","threshold":safety_thresh},{"category":"HARM_CATEGORY_DANGEROUS","threshold":safety_thresh}],}
        
        response = palm.generate_text(**PaLM_settings, prompt=prompt)


        if response and response.result:
            if isinstance(response.result, (str, bytes)):
                response_valid = check_and_redo_JSON(response, PaLM_settings, logger, version)
            else:
                response_valid = {}
        else:
            response_valid = {}

        logger.info(f'Candidate JSON\n{response.result}')
        return response_valid

def check_and_redo_JSON(response, PaLM_settings, logger, version):
    try:
        response_valid = json.loads(response.result)
        logger.info(f'Response --- First call passed')
        return response_valid
    except JSONDecodeError:

        try:
            response_valid = json.loads(response.result.strip('```').replace('json\n', '', 1).replace('json', '', 1))
            logger.info(f'Response --- Manual removal of ```json succeeded')
            return response_valid
        except:
            logger.info(f'Response --- First call failed. Redo...')
            Prompt = PromptCatalog() 
            if version == 'v1':
                prompt_redo = Prompt.prompt_palm_redo_v1(response.result)
            elif version == 'v2':
                prompt_redo = Prompt.prompt_palm_redo_v2(response.result)

            # prompt_redo = PROMPT_PaLM_Redo(response.result)
            try:
                response = palm.generate_text(**PaLM_settings, prompt=prompt_redo)
                response_valid = json.loads(response.result)
                logger.info(f'Response --- Second call passed')
                return response_valid
            except JSONDecodeError:
                logger.info(f'Response --- Second call failed. Final redo. Temperature changed to 0.05')
                try:
                    PaLM_settings["temperature"]=0.05
                    response = palm.generate_text(**PaLM_settings, prompt=prompt_redo)
                    response_valid = json.loads(response.result)
                    logger.info(f'Response --- Third call passed')
                    return response_valid
                except JSONDecodeError:
                    return None
            

def create_OCR_analog_for_input(domain_knowledge_example):
    in_list = []
    out_list = []
    # Iterate over the domain_knowledge_example (list of dictionaries)
    for row_dict in domain_knowledge_example:
        # Convert the dictionary to a JSON string and add it to the out_list
        domain_knowledge_example_string = json.dumps(row_dict)
        out_list.append(domain_knowledge_example_string)

        # Create a single string from all values in the row_dict
        row_text = '||'.join(str(v) for v in row_dict.values())

        # Split the row text by '||', shuffle the parts, and then re-join with a single space
        parts = row_text.split('||')
        random.shuffle(parts)
        shuffled_text = ' '.join(parts)

        # Add the shuffled_text to the in_list
        in_list.append(shuffled_text)
    return in_list, out_list


def strip_problematic_chars(s):
    return ''.join(c for c in s if c.isprintable())


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens