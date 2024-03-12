import os, inspect, sys, shutil



class AllOptions():
    a_llm = [
        "GPT 4 Turbo 1106-preview",
        "GPT 4 Turbo 0125-preview",
        'GPT 4',
        'GPT 4 32k',
        'GPT 3.5',
        'GPT 3.5 Instruct',

        'Azure GPT 3.5',
        'Azure GPT 3.5 Instruct',
        'Azure GPT 4',
        'Azure GPT 4 Turbo 1106-preview',
        'Azure GPT 4 Turbo 0125-preview',
        'Azure GPT 4 32k',

        'PaLM 2 text-bison@001',
        'PaLM 2 text-bison@002',
        'PaLM 2 text-unicorn@001',
        'Gemini Pro',

        'Mistral Small',
        'Mistral Medium',
        'Mistral Large',
        'Open Mixtral 8x7B',
        'Open Mistral 7B',

        'LOCAL Mixtral 8x7B Instruct v0.1',
        'LOCAL Mistral 7B Instruct v0.2',

        'LOCAL CPU Mistral 7B Instruct v0.2 GGUF',
        ]
    
    a_prompt_version = [
        'SLTPvA_long.yaml',
        'SLTPvA_medium.yaml',
        'SLTPvA_short.yaml',
        'SLTPvB_long.yaml',
        'SLTPvB_medium.yaml',
        'SLTPvB_short.yaml',
        'SLTPvB_minimal.yaml',
    ]

    a_LM2 = [False,] # [True, False]
    a_do_use_trOCR = [False,] # [True, False]
    a_trocr_path = ["microsoft/trocr-large-handwritten",]
    a_ocr_option = [
        'hand',
        'normal',
        'CRAFT',
        'LLaVA',
        ['hand','CRAFT'],
        ['hand','LLaVA'],
        ]
    a_llava_option = ["llava-v1.6-mistral-7b", 
                      "llava-v1.6-34b", 
                      "llava-v1.6-vicuna-13b", 
                      "llava-v1.6-vicuna-7b",]
    a_llava_bit = ["full", "4bit",]
    a_double_ocr = [True, False]




class Options_permute_llms_to_investigate_determinism_at_restrictive_settings():
    a_llm = [
        # "GPT 4 Turbo 1106-preview",
        # "GPT 4 Turbo 0125-preview",
        # 'GPT 4',
        # # 'GPT 4 32k',
        # 'GPT 3.5 Turbo',
        # 'GPT 3.5 Instruct',

        'Azure GPT 3.5 Turbo',
        'Azure GPT 3.5 Instruct',
        'Azure GPT 4',
        'Azure GPT 4 Turbo 1106-preview',
        'Azure GPT 4 Turbo 0125-preview',
        # 'Azure GPT 4 32k',

        'PaLM 2 text-bison@001',
        'PaLM 2 text-bison@002',
        'PaLM 2 text-unicorn@001',
        'Gemini Pro',

        'Mistral Small',
        'Mistral Medium',
        'Mistral Large',
        # 'Open Mixtral 8x7B',
        'Open Mistral 7B',

        # 'LOCAL Mixtral 8x7B Instruct v0.1',
        # 'LOCAL Mistral 7B Instruct v0.2',

        # 'LOCAL CPU Mistral 7B Instruct v0.2 GGUF',
        ]
    
    a_prompt_version = [
        # 'SLTPvA_long.yaml',
        # 'SLTPvA_short.yaml',
        'SLTPvB_long.yaml',
        'SLTPvB_short.yaml',
        'SLTPvB_minimal.yaml',
    ]
    a_double_ocr = [True, False]

    ### BELOW ARE STATIC
    a_LM2 = [False,]
    # a_do_use_trOCR = [True, False]
    a_do_use_trOCR = [False,]
    # a_trocr_path = ["microsoft/trocr-large-handwritten","microsoft/trocr-base-handwritten",]
    a_trocr_path = ["microsoft/trocr-large-handwritten",]
    a_ocr_option = ['hand',]
    a_llava_option = ["llava-v1.6-mistral-7b",]
    a_llava_bit = ["full",]


class Options_permute_llms_to_sweep_temperature_and_topP_for_GPT4_0125():
    a_llm = [
        # 'Azure GPT 4 Turbo 0125-preview', #test 1
        'Azure GPT 4',
        ]
    
    a_prompt_version = [
        # 'SLTPvA_long.yaml',
        # 'SLTPvA_short.yaml',
        'SLTPvB_long.yaml',
        'SLTPvB_short.yaml',
        # 'SLTPvB_minimal.yaml',
    ]
    a_double_ocr = [True, False]

    ### BELOW ARE STATIC
    a_LM2 = [False,]
    # a_do_use_trOCR = [True, False]
    a_do_use_trOCR = [False,]
    # a_trocr_path = ["microsoft/trocr-large-handwritten","microsoft/trocr-base-handwritten",]
    a_trocr_path = ["microsoft/trocr-large-handwritten",]
    a_ocr_option = ['hand',]
    a_llava_option = ["llava-v1.6-mistral-7b",]
    a_llava_bit = ["full",]


class Options_permute_llms_to_sweep_temperature_and_topP_for_google():
    a_llm = [
        'PaLM 2 text-bison@001',
        'PaLM 2 text-bison@002',
        'Gemini Pro',
        ]
    
    a_prompt_version = [
        'SLTPvB_long.yaml',
        'SLTPvB_short.yaml',
    ]
    a_double_ocr = [True, False]

    ### BELOW ARE STATIC
    a_LM2 = [False,]
    a_do_use_trOCR = [False,] # [True, False]
    a_trocr_path = ["microsoft/trocr-large-handwritten",]
    a_ocr_option = ['hand',]
    a_llava_option = ["llava-v1.6-mistral-7b",]
    a_llava_bit = ["full",]

if __name__ == '__main__':
    pass