import os, sys, inspect, json, time

# currentdir = os.path.dirname(os.path.abspath(
#     inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

# from prompts import PROMPT_PaLM_UMICH_skeleton_all_asia, PROMPT_PaLM_OCR_Organized, PROMPT_PaLM_Redo
# from LLM_PaLM import create_OCR_analog_for_input, num_tokens_from_string

'''
https://docs.ai21.com/docs/python-sdk-with-amazon-bedrock


https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/falcon-llms-in-azure-machine-learning/ba-p/3876847
https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/huggingface/inference/text-generation-streaming/text-generation-streaming-online-endpoint.ipynb
https://ml.azure.com/registries/HuggingFace/models/tiiuae-falcon-40b-instruct/version/12?tid=e66e77b4-5724-44d7-8721-06df160450ce#overview
https://azure.microsoft.com/en-us/products/machine-learning/
'''



# from azure.ai.ml import MLClient
# from azure.identity import (
#     DefaultAzureCredential,
#     InteractiveBrowserCredential,
#     ClientSecretCredential,
# )
# from azure.ai.ml.entities import AmlCompute

# try:
#     credential = DefaultAzureCredential()
#     credential.get_token("https://management.azure.com/.default")
# except Exception as ex:
#     credential = InteractiveBrowserCredential()

# # connect to a workspace
# workspace_ml_client = None
# try:
#     workspace_ml_client = MLClient.from_config(credential)
#     subscription_id = workspace_ml_client.subscription_id
#     workspace = workspace_ml_client.workspace_name
#     resource_group = workspace_ml_client.resource_group_name
# except Exception as ex:
#     print(ex)
#     # Enter details of your workspace
#     subscription_id = "<SUBSCRIPTION_ID>"
#     resource_group = "<RESOURCE_GROUP>"
#     workspace = "<AML_WORKSPACE_NAME>"
#     workspace_ml_client = MLClient(
#         credential, subscription_id, resource_group, workspace
#     )
# # Connect to the HuggingFaceHub registry
# registry_ml_client = MLClient(credential, registry_name="HuggingFace")
# print(registry_ml_client)

'''
def OCR_to_dict_Falcon(logger, OCR, VVE):
    # Find a similar example from the domain knowledge
    domain_knowledge_example = VVE.query_db(OCR, 4)
    similarity = VVE.get_similarity()
    domain_knowledge_example_string = json.dumps(domain_knowledge_example)

    try:
        logger.info(f'Length of OCR raw -- {len(OCR)}')
    except:
        print(f'Length of OCR raw -- {len(OCR)}')

    # Create input: output: for Falcon
    # Assuming Falcon requires a similar structure as PaLM
    in_list, out_list = create_OCR_analog_for_input(domain_knowledge_example)

    # Construct the prompt for Falcon
    # Adjust this based on Falcon's requirements
    # prompt = PROMPT_Falcon_skeleton(OCR, in_list, out_list)
    prompt = PROMPT_PaLM_UMICH_skeleton_all_asia(OCR, in_list, out_list) # must provide examples to PaLM differently than for chatGPT, at least 2 examples


    nt = num_tokens_from_string(prompt, "falcon_model_name")  # Replace "falcon_model_name" with the appropriate model name for Falcon
    try:
        logger.info(f'Prompt token length --- {nt}')
    except:
        print(f'Prompt token length --- {nt}')

    # Assuming Falcon has a similar API structure as PaLM
    # Adjust the settings based on Falcon's requirements
    Falcon_settings = {
        'model': 'models/falcon_model_name',  # Replace with the appropriate model name for Falcon
        'temperature': 0,
        'candidate_count': 1,
        'top_k': 40,
        'top_p': 0.95,
        'max_output_tokens': 8000,
        'stop_sequences': [],
        # Add any other required settings for Falcon
    }

    # Send the prompt to Falcon for inference
    # Adjust the API call based on Falcon's requirements
    response = falcon.generate_text(**Falcon_settings, prompt=prompt)

    # Process the response from Falcon
    if response and response.result:
        if isinstance(response.result, (str, bytes)):
            response_valid = check_and_redo_JSON(response, Falcon_settings, logger)
        else:
            response_valid = {}
    else:
        response_valid = {}

    return response_valid
'''