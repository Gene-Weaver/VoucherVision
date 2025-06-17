class ModelMaps:
    PROMPTS_THAT_NEED_DOMAIN_KNOWLEDGE = ['Version 1', 'Version 1 PaLM 2']
    COLORS_EXPENSE_REPORT = {
        'GPT_4': '#32CD32',  # Lime Green
        'GPT_3_5': '#008000',  # Green
        'GPT_4_TURBO_2024_04_09': '#32CD32',  # Lime Green
        'GPT_4o_2024_05_13': '#3CB371',  # Lime Green gpt-4o-2024-05-13
        'GPT_4o_2024_08_06': '#3CB371',  # Lime Green gpt-4o-2024-05-13
        'GPT_4o_mini_2024_07_18': '#115730',  # Lime Green gpt-4o-2024-05-13
        'GPT_3_5_INSTRUCT': '#3CB371',  # Medium Sea Green
        'GPT_4_TURBO_1106': '#228B22',  # Forest Green
        'GPT_4_TURBO_0125': '#228B22',  # Forest Green
        'GPT_4_32K': '#006400',  # Dark Green

        # 'PALM2_TB_1': '#87CEEB',  # Sky Blue
        'PALM2_TB_2': '#1E90FF',  # Dodger Blue
        'PALM2_TU_1': '#0000FF',  # Blue
        'GEMINI_1_5_FLASH_8B': '#1E00FF',  # 
        'GEMINI_1_5_FLASH': '#1E00FF',  # gemini-1.5-flash
        'GEMINI_1_5_PRO': '#1E00FF',  # gemini-1.5-pro
        'GEMINI_2_0_FLASH': '#1E00FF',  # gemini-1.5-pro
        'GEMINI_2_5_FLASH': '#1E00FF',  # gemini-1.5-pro
        'GEMINI_2_5_PRO': '#1E00FF',  # gemini-1.5-pro

        'AZURE_GPT_4': '#800080',  # Purple
        'AZURE_GPT_4o': '#800080',  # Purple
        'AZURE_GPT_4o_mini': '#800080',  # Purple
        # 'AZURE_GPT_4_TURBO_1106': '#9370DB',  # Medium Purple
        # 'AZURE_GPT_4_TURBO_0125': '#9370DB',  # Medium Purple
        # 'AZURE_GPT_4_32K': '#8A2BE2',  # Blue Violet
        # 'AZURE_GPT_3_5_INSTRUCT': '#9400D3',  # Dark Violet
        # 'AZURE_GPT_3_5': '#9932CC',  # Dark Orchid


        'Hyperbolic_VLM_Pixtral_12B': '#FF6347',
        'Hyperbolic_VLM_Qwen2_VL_7B_Instruct':'#FF6347',
        'Hyperbolic_VLM_Qwen2_VL_72B_Instruct':'#FF6347',
        # 'Hyperbolic_VLM_Llama_3_2_90B_Vision_Instruct':'#FF6347',
        'Hyperbolic_LLM_Llama_3_1_8B':'#FF6347',
        'Hyperbolic_LLM_Llama_3_1_70B':'#FF6347',
        'Hyperbolic_LLM_Llama_3_70B':'#FF6347',
        # 'Hyperbolic_LLM_Hermes_3_70B':'#FF6347',
        'Hyperbolic_LLM_Llama_3_1_405B':'#FF6347',
        'Hyperbolic_LLM_Llama_3_1_405B_FP8':'#FF6347',
        'Hyperbolic_LLM_DeepSeek_V2_5':'#FF6347',
        'Hyperbolic_LLM_Qwen_2_5_72B':'#FF6347',
        'Hyperbolic_LLM_Llama_3_2_3B':'#FF6347',
        'Hyperbolic_LLM_Qwen_2_5_Coder_32B':'#FF6347',
        'Hyperbolic_LLM_QwQ_32B_Preview':'#FF6347',

        'OPEN_MISTRAL_7B': '#800000',  # Maroon
        'OPEN_MIXTRAL_8X7B': '#800000',  # Maroon
        'OPEN_MIXTRAL_8X22B': '#800000',  # Maroon
        'MISTRAL_NEMO': '#800000',  # Maroon
        'PIXTRAL_12B': '#800000',  # Maroon
        'MISTRAL_LARGE': '#800000',  # Maroon
        'PIXTRAL_LARGE': '#800000',  # Maroon
        'MISTRAL_SMALL': '#800000',  # Maroon
        'MINISTRAL_8B_24_10': '#800000',  # Maroon
        'MINISTRAL_3B_24_10': '#800000',  # Maroon

        'LOCAL_MIXTRAL_8X7B_INSTRUCT_V01': '#000000',  # Black
        'LOCAL_MISTRAL_7B_INSTRUCT_V02': '#4a4a4a',  # Gray
        #  mistralai/Mistral-Nemo-Instruct-2407
        'LOCAL_MISTRAL_NEMO_INSTRUCT_2407': '#000000',  # Black

        'LOCAL_CPU_MISTRAL_7B_INSTRUCT_V02_GGUF': '#bababa',  # Gray

        'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05': '#bababa',  # Gray
    }



    ################################
    # To keep support, but hide from user, just comment out the entry in this section
    ################################
    MODELS_OPENAI = [
                    'GPT 4o 2024-08-06',
                    'GPT 4o 2024-05-13', #GPT_4o_2024_05_13
                    'GPT 4o mini 2024-07-18',
                    'GPT 4 Turbo 2024-04-09',#GPT_4_TURBO_2024_04_09

                    'GPT 4',
                    'GPT 4 32k',
                    'GPT 4 Turbo 0125-preview',
                    'GPT 4 Turbo 1106-preview',
                    'GPT 3.5 Turbo',
                    'GPT 3.5 Instruct',
                    ]
    

    MODELS_OPENAI_AZURE = [
                    'Azure GPT 4o',
                    'Azure GPT 4o mini',
                    'Azure GPT 4',
                    #  'Azure GPT 4 32k',
                    #  'Azure GPT 4 Turbo 0125-preview',
                    #  'Azure GPT 4 Turbo 1106-preview',
                    #  'Azure GPT 3.5 Turbo',
                    #  'Azure GPT 3.5 Instruct',
                    ]
    
    MODELS_GOOGLE = [
                    # 'PaLM 2 text-bison@001',
                     'Gemini 2.5 Pro',
                     'Gemini 2.5 Flash',
                     'Gemini 2.0 Flash',
                     'Gemini 1.5 Pro',
                     'Gemini 1.5 Flash',
                     'Gemini 1.5 Flash 8B',
                     'PaLM 2 text-bison@002',
                     'PaLM 2 text-unicorn@001',
                     ]

    MODELS_MISTRAL = ['Open Mistral 7B',
                        'Open Mixtral 8x7B',
                        'Open Mixtral 8x22B',
                        'Mixtral NeMo',
                        'Pixtral 12B',
                        'Mistral Large',
                        'Pixtral Large',
                        'Mistral Small',
                        'Ministral 8B',
                        'Ministral 3B',
                      ]
    
    MODELS_HYPERBOLIC = [
                        # 'Hyperbolic VLM Pixtral 12B',
                        # 'Hyperbolic VLM Qwen2 VL 7B Instruct',
                        'Hyperbolic VLM Qwen2 VL 72B Instruct', # PASS
                        # 'Hyperbolic VLM Llama 3 2 90B Vision Instruct', # Not available
                        'Hyperbolic LLM Llama 3.2-3B-Instruct', # PASS
                        'Hyperbolic LLM Llama 3.1-8B-Instruct', # PASS
                        'Hyperbolic LLM Llama 3.1-70B-Instruct', # PASS
                        # 'Hyperbolic LLM Llama 3-70B',
                        # 'Hyperbolic LLM Hermes 3-70B', # Not available
                        # 'Hyperbolic LLM Llama 3.1-405B',
                        # 'Hyperbolic LLM Llama 3.1-405B-FP8',
                        # 'Hyperbolic LLM DeepSeek-V2.5',
                        'Hyperbolic LLM Qwen 2.5-72B-Instruct', # PASS
                        'Hyperbolic LLM Qwen 2.5-Coder-32B-Instruct',# PASS
                        # 'Hyperbolic LLM QwQ-32B-Preview',
                    ]

    MODELS_LOCAL = ['LOCAL Mistral Nemo Instruct 2407', 
                    'LOCAL Mixtral 8x7B Instruct v0.1',
                    'LOCAL Mistral 7B Instruct v0.2',
                    'LOCAL CPU Mistral 7B Instruct v0.2 GGUF',
                    'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05']

    MODELS_GUI_DEFAULT = 'Gemini 2.0 Flash' #'Gemini 1.5 Flash' #'Azure GPT 4' # 'GPT 4 Turbo 1106-preview'

    MODEL_FAMILY = {
                    'OpenAI': MODELS_OPENAI,
                    'OpenAI Azure': MODELS_OPENAI_AZURE,
                    'Google': MODELS_GOOGLE, 
                    'Hyperbolic': MODELS_HYPERBOLIC, 
                    'Mistral': MODELS_MISTRAL, 
                    'Local': MODELS_LOCAL}

    version_mapping_cost = {
        'GPT 4 32k': 'GPT_4_32K',
        'GPT 4': 'GPT_4',
        'GPT 4o 2024-05-13': 'GPT_4o_2024_05_13',
        'GPT 4o 2024-08-06': 'GPT_4o_2024_08_06',

        'GPT 4o mini 2024-07-18': 'GPT_4o_mini_2024_07_18',
        'GPT 4 Turbo 2024-04-09': 'GPT_4_TURBO_2024_04_09',
        'GPT 4 Turbo 0125-preview': 'GPT_4_TURBO_0125',
        'GPT 4 Turbo 1106-preview': 'GPT_4_TURBO_1106',
        'GPT 3.5 Instruct': 'GPT_3_5_INSTRUCT',
        'GPT 3.5 Turbo': 'GPT_3_5',

        # 'Azure GPT 4 32k': 'AZURE_GPT_4_32K',
        'Azure GPT 4': 'AZURE_GPT_4',
        'Azure GPT 4o': 'AZURE_GPT_4o',
        'Azure GPT 4o mini': 'AZURE_GPT_4o_mini',
        # 'Azure GPT 4 Turbo 0125-preview': 'AZURE_GPT_4_TURBO_0125',
        # 'Azure GPT 4 Turbo 1106-preview': 'AZURE_GPT_4_TURBO_1106',
        # 'Azure GPT 3.5 Instruct': 'AZURE_GPT_3_5_INSTRUCT',
        # 'Azure GPT 3.5 Turbo': 'AZURE_GPT_3_5',

        'Gemini 2.5 Pro': 'GEMINI_2_5_PRO',  # gemini-1.5-pro
        'Gemini 2.5 Flash': 'GEMINI_2_5_FLASH',  # gemini-1.5-pro
        'Gemini 2.0 Flash': 'GEMINI_2_0_FLASH',  # gemini-1.5-pro
        'Gemini 1.5 Pro': 'GEMINI_1_5_PRO',  # gemini-1.5-pro
        'Gemini 1.5 Flash': 'GEMINI_1_5_FLASH',  # gemini-1.5-flash
        'Gemini 1.5 Flash 8B': 'GEMINI_1_5_FLASH_8B',

        'PaLM 2 text-unicorn@001': 'PALM2_TU_1',
        # 'PaLM 2 text-bison@001': 'PALM2_TB_1',
        'PaLM 2 text-bison@002': 'PALM2_TB_2',

        'Open Mistral 7B':  'OPEN_MISTRAL_7B',
        'Open Mixtral 8x7B':  'OPEN_MIXTRAL_8X7B',
        'Open Mixtral 8x22B':  'OPEN_MIXTRAL_8X22B',
        'Mixtral NeMo':  'MISTRAL_NEMO',
        'Pixtral 12B':  'PIXTRAL_12B',
        'Mistral Large':  'MISTRAL_LARGE',
        'Pixtral Large':  'PIXTRAL_LARGE',
        'Mistral Small':  'MISTRAL_SMALL',
        'Ministral 8B':  'MINISTRAL_8B_24_10',
        'Ministral 3B':  'MINISTRAL_3B_24_10',

        'Hyperbolic VLM Pixtral 12B': 'Hyperbolic_VLM_Pixtral_12B',
        'Hyperbolic VLM Qwen2 VL 7B Instruct': 'Hyperbolic_VLM_Qwen2_VL_7B_Instruct',
        'Hyperbolic VLM Qwen2 VL 72B Instruct': 'Hyperbolic_VLM_Qwen2_VL_72B_Instruct',
        # 'Hyperbolic VLM Llama 3 2 90B Vision Instruct': 'Hyperbolic_VLM_Llama_3_2_90B_Vision_Instruct',

        'Hyperbolic LLM Qwen 2.5-72B-Instruct': 'Hyperbolic_LLM_Qwen_2_5_72B',
        'Hyperbolic LLM Qwen 2.5-Coder-32B-Instruct': 'Hyperbolic_LLM_Qwen_2_5_Coder_32B',
        'Hyperbolic LLM Llama 3.2-3B-Instruct': 'Hyperbolic_LLM_Llama_3_2_3B',
        'Hyperbolic LLM Llama 3.1-405B': 'Hyperbolic_LLM_Llama_3_1_405B',
        'Hyperbolic LLM Llama 3.1-405B-FP8': 'Hyperbolic_LLM_Llama_3_1_405B_FP8',
        'Hyperbolic LLM Llama 3.1-8B-Instruct': 'Hyperbolic_LLM_Llama_3_1_8B',
        'Hyperbolic LLM Llama 3.1-70B-Instruct': 'Hyperbolic_LLM_Llama_3_1_70B',
        'Hyperbolic LLM Llama 3-70B': 'Hyperbolic_LLM_Llama_3_70B',
        # 'Hyperbolic LLM Hermes 3-70B': 'Hyperbolic_LLM_Hermes_3_70B',
        'Hyperbolic LLM DeepSeek-V2.5': 'Hyperbolic_LLM_DeepSeek_V2_5',
        'Hyperbolic LLM QwQ-32B-Preview': 'Hyperbolic_LLM_QwQ_32B_Preview',

        'LOCAL Mistral Nemo Instruct 2407': 'LOCAL_MISTRAL_NEMO_INSTRUCT_2407', 
        'LOCAL Mixtral 8x7B Instruct v0.1': 'LOCAL_MIXTRAL_8X7B_INSTRUCT_V01',
        'LOCAL Mistral 7B Instruct v0.2': 'LOCAL_MISTRAL_7B_INSTRUCT_V02',

        'LOCAL CPU Mistral 7B Instruct v0.2 GGUF': 'LOCAL_CPU_MISTRAL_7B_INSTRUCT_V02_GGUF',

        'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05': 'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05',
    }

    @classmethod
    def get_version_has_key(cls, key, has_key_openai, has_key_azure_openai, has_key_google_application_credentials, has_key_mistral, has_key_hyperbolic):
        # Define the mapping for 'has_key' values
        version_has_key = {
            'GPT 4 Turbo 2024-04-09': has_key_openai,
            'GPT 4 Turbo 1106-preview': has_key_openai,
            'GPT 4 Turbo 0125-preview': has_key_openai,
            'GPT 4':  has_key_openai,
            'GPT 4o 2024-05-13': has_key_openai, 
            'GPT 4o 2024-08-06': has_key_openai, 
            'GPT 4o mini 2024-07-18': has_key_openai, 
            'GPT 4 32k':  has_key_openai,
            'GPT 3.5 Turbo':  has_key_openai,
            'GPT 3.5 Instruct':  has_key_openai,

            # 'Azure GPT 3.5 Turbo': has_key_azure_openai,
            # 'Azure GPT 3.5 Instruct': has_key_azure_openai,
            'Azure GPT 4': has_key_azure_openai,
            'Azure GPT 4o': has_key_azure_openai,
            'Azure GPT 4o mini': has_key_azure_openai,
            # 'Azure GPT 4 Turbo 1106-preview': has_key_azure_openai,
            # 'Azure GPT 4 Turbo 0125-preview': has_key_azure_openai,
            # 'Azure GPT 4 32k': has_key_azure_openai,

            # 'PaLM 2 text-bison@001':  has_key_google_application_credentials,

            'Gemini 2.5 Pro':  has_key_google_application_credentials,
            'Gemini 2.5 Flash':  has_key_google_application_credentials,
            'Gemini 2.0 Flash':  has_key_google_application_credentials,
            'Gemini 1.5 Pro':  has_key_google_application_credentials,
            'Gemini 1.5 Flash':  has_key_google_application_credentials,
            'Gemini 1.5 Flash 8B':  has_key_google_application_credentials,
            'PaLM 2 text-bison@002':  has_key_google_application_credentials,
            'PaLM 2 text-unicorn@001':  has_key_google_application_credentials,

            'Open Mistral 7B':  has_key_mistral,
            'Open Mixtral 8x7B':  has_key_mistral,
            'Open Mixtral 8x22B':  has_key_mistral,
            'Mixtral NeMo':  has_key_mistral,
            'Pixtral 12B':  has_key_mistral,
            'Mistral Large':  has_key_mistral,
            'Pixtral Large':  has_key_mistral,
            'Mistral Small':  has_key_mistral,
            'Ministral 8B':  has_key_mistral,
            'Ministral 3B':  has_key_mistral,

            'Hyperbolic VLM Pixtral 12B': has_key_hyperbolic,
            'Hyperbolic VLM Qwen2 VL 7B Instruct': has_key_hyperbolic,
            'Hyperbolic VLM Qwen2 VL 72B Instruct': has_key_hyperbolic,
            'Hyperbolic VLM Llama 3 2 90B Vision Instruct': has_key_hyperbolic,

            'Hyperbolic LLM Qwen 2.5-72B-Instruct': has_key_hyperbolic,
            'Hyperbolic LLM Qwen 2.5-Coder-32B-Instruct': has_key_hyperbolic,
            'Hyperbolic LLM Llama 3.2-3B-Instruct': has_key_hyperbolic,
            'Hyperbolic LLM Llama 3.1-405B': has_key_hyperbolic,
            'Hyperbolic LLM Llama 3.1-405B-FP8': has_key_hyperbolic,
            'Hyperbolic LLM Llama 3.1-8B-Instruct': has_key_hyperbolic,
            'Hyperbolic LLM Llama 3.1-70B-Instruct': has_key_hyperbolic,
            'Hyperbolic LLM Llama 3-70B': has_key_hyperbolic,
            # 'Hyperbolic LLM Hermes 3-70B': has_key_hyperbolic,
            'Hyperbolic LLM DeepSeek-V2.5': has_key_hyperbolic,
            'Hyperbolic LLM QwQ-32B-Preview': has_key_hyperbolic,

            'LOCAL Mistral Nemo Instruct 2407': True, 
            'LOCAL Mixtral 8x7B Instruct v0.1':  True,
            'LOCAL Mistral 7B Instruct v0.2':  True,

            'LOCAL CPU Mistral 7B Instruct v0.2 GGUF':  True,

            'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05': True
        }
        return version_has_key.get(key)

    @classmethod
    def get_version_mapping_is_azure(cls, key):
        version_mapping_is_azure = {
            'GPT 4o 2024-05-13': False, 
            'GPT 4o 2024-08-06': False, 
            'GPT 4o mini 2024-07-18': False, 
            'GPT 4 Turbo 2024-04-09': False,
            'GPT 4 Turbo 1106-preview': False,
            'GPT 4 Turbo 0125-preview': False,
            'GPT 4': False,
            'GPT 4 32k':  False,
            'GPT 3.5 Turbo':  False,
            'GPT 3.5 Instruct':  False,
                                       
            # 'Azure GPT 3.5 Turbo': True,
            # 'Azure GPT 3.5 Instruct': True,
            'Azure GPT 4': True,
            'Azure GPT 4o': True,
            'Azure GPT 4o mini': True,
            # 'Azure GPT 4 Turbo 1106-preview': True,
            # 'Azure GPT 4 Turbo 0125-preview': True,
            # 'Azure GPT 4 32k': True,

            # 'PaLM 2 text-bison@001':  False,
            'Gemini 2.5 Pro':  False,
            'Gemini 2.5 Flash':  False,
            'Gemini 2.0 Flash':  False,
            'Gemini 1.5 Pro':  False,
            'Gemini 1.5 Flash':  False,
            'Gemini 1.5 Flash 8B':  False,
            'PaLM 2 text-bison@002':  False,
            'PaLM 2 text-unicorn@001':  False,

            'Open Mistral 7B':  False,
            'Open Mixtral 8x7B':  False,
            'Open Mixtral 8x22B':  False,
            'Mixtral NeMo':  False,
            'Pixtral 12B':  False,
            'Mistral Large':  False,
            'Pixtral Large':  False,
            'Mistral Small':  False,
            'Ministral 8B':  False,
            'Ministral 3B':  False,

            'Hyperbolic VLM Pixtral 12B':  False,
            'Hyperbolic VLM Qwen2 VL 7B Instruct':  False,
            'Hyperbolic VLM Qwen2 VL 72B Instruct':  False,
            # 'Hyperbolic VLM Llama 3 2 90B Vision Instruct':  False,

            'Hyperbolic LLM Qwen 2.5-72B-Instruct':  False,
            'Hyperbolic LLM Qwen 2.5-Coder-32B-Instruct':  False,
            'Hyperbolic LLM Llama 3.2-3B-Instruct':  False,
            'Hyperbolic LLM Llama 3.1-405B':  False,
            'Hyperbolic LLM Llama 3.1-405B-FP8': False,
            'Hyperbolic LLM Llama 3.1-8B-Instruct':  False,
            'Hyperbolic LLM Llama 3.1-70B-Instruct':  False,
            'Hyperbolic LLM Llama 3-70B':  False,
            # 'Hyperbolic LLM Hermes 3-70B':  False,
            'Hyperbolic LLM DeepSeek-V2.5':  False,
            'Hyperbolic LLM QwQ-32B-Preview':  False,

            'LOCAL Mistral Nemo Instruct 2407':  False, 
            'LOCAL Mixtral 8x7B Instruct v0.1':  False,
            'LOCAL Mistral 7B Instruct v0.2':  False,

            'LOCAL CPU Mistral 7B Instruct v0.2 GGUF':  False,

            'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05': False
        }
        return version_mapping_is_azure.get(key)

    @classmethod
    def get_API_name(cls, key):
        
        ### OpenAI
        if key == 'GPT_3_5':
            return 'gpt-3.5-turbo-0125' #'gpt-3.5-turbo-1106'
        
        elif key == 'GPT_3_5_INSTRUCT':
            return 'gpt-3.5-turbo-instruct'
        
        elif key == 'GPT_4':
            return 'gpt-4'

        elif key == 'GPT_4_32K':
            return 'gpt-4-32k'

        elif key == 'GPT_4o_2024_05_13':
            return 'gpt-4o-2024-05-13'
        
        elif key == 'GPT_4o_2024_08_06':
            return 'gpt-4o-2024-08-06'

        elif key == 'GPT_4o_mini_2024_07_18':
            return 'gpt-4o-mini-2024-07-18'
        
        elif key == 'GPT_4_TURBO_2024_04_09':
            return 'gpt-4-turbo-2024-04-09'

        elif key == 'GPT_4_TURBO_1106':
            return 'gpt-4-1106-preview'
        
        elif key == 'GPT_4_TURBO_0125':
            return 'gpt-4-0125-preview'
        
        ### Azure
        # elif key == 'AZURE_GPT_3_5':
        #     return 'gpt-35-turbo-0125'

        # elif key == 'AZURE_GPT_3_5_INSTRUCT':
        #     return 'gpt-35-turbo-instruct'
        
        elif key == 'AZURE_GPT_4':
            return 'gpt-4'
        elif key == 'AZURE_GPT_4o':
            return 'gpt-4o'
        elif key == 'AZURE_GPT_4o_mini':
            return 'gpt-4o-mini'

        # elif key == 'AZURE_GPT_4_TURBO_1106':
        #     return 'gpt-4-1106-preview'
        
        # elif key == 'AZURE_GPT_4_TURBO_0125':
        #     return 'gpt-4-0125-preview'
        
        # elif key == 'AZURE_GPT_4_32K':
        #     return 'gpt-4-32k'
        
        ### Google
        # elif key == 'PALM2_TB_1':
        #     return 'text-bison@001'
        
        elif key == 'PALM2_TB_2':
            return 'text-bison@002'
        
        elif key == 'PALM2_TU_1':
            return 'text-unicorn@001'
        
        elif key == 'GEMINI_1_5_FLASH_8B':
            return 'gemini-1.5-flash-8b' 

        elif key == 'GEMINI_1_5_FLASH':
            return 'gemini-1.5-flash' 

        elif key == 'GEMINI_1_5_PRO':
            return 'gemini-1.5-pro' 
        
        elif key == 'GEMINI_2_0_FLASH':
            return 'gemini-2.0-flash'
        
        elif key == 'GEMINI_2_5_FLASH':
            return 'gemini-2.5-flash' # TODO UPDATE AS NEEDED
        
        elif key == 'GEMINI_2_5_PRO':
            return 'gemini-2.5-pro' # TODO UPDATE AS NEEDED

        # elif key == 'GEMINI_2_0_PRO':
        #     return 'gemini-2.0-pro' 
        
        ### Mistral 
        elif key == 'OPEN_MISTRAL_7B':
            return 'open-mistral-7b'
        elif key == 'OPEN_MIXTRAL_8X7B':
            return 'open-mixtral-8x7b'
        elif key == 'OPEN_MIXTRAL_8X22B':
            return 'open-mixtral-8x22b'
        elif key == 'MISTRAL_NEMO':
            return 'mistral-nemo'
        elif key == 'PIXTRAL_12B':
            return 'pixtral-12b'
        
        elif key == 'MISTRAL_LARGE':
            return 'mistral-large-latest'
        elif key == 'PIXTRAL_LARGE':
            return 'pixtral-large-latest'
        elif key == 'MISTRAL_SMALL':
            return 'mistral-small-latest'
        elif key == 'MINISTRAL_8B_24_10':
            return 'ministral-8b-latest'
        elif key == 'MINISTRAL_3B_24_10':
            return 'ministral-3b-latest'

        elif key == 'Hyperbolic_VLM_Pixtral_12B': 
            return 'mistralai/Pixtral-12B-2409'
        elif key == 'Hyperbolic_VLM_Qwen2_VL_7B_Instruct': 
            return 'Qwen/Qwen2-VL-7B-Instruct'
        elif key == 'Hyperbolic_VLM_Qwen2_VL_72B_Instruct': 
            return 'Qwen/Qwen2-VL-72B-Instruct'
        
        # elif key == 'Hyperbolic_VLM_Llama_3_2_90B_Vision_Instruct': 
        #     return 'meta-llama/Llama-3.2-90B-Vision-Instruct'

        elif key == 'Hyperbolic_LLM_Qwen_2_5_72B': 
            return 'Qwen/Qwen2.5-72B-Instruct'
        elif key == 'Hyperbolic_LLM_QwQ_32B_Preview': 
            return 'Qwen/QwQ-32B-Preview'
        elif key == 'Hyperbolic_LLM_Qwen_2_5_Coder_32B': 
            return 'Qwen/Qwen2.5-Coder-32B-Instruct'
        elif key == 'Hyperbolic_LLM_Llama_3_2_3B': 
            return 'meta-llama/Llama-3.2-3B-Instruct'
        elif key == 'Hyperbolic_LLM_Llama_3_1_405B': 
            return 'meta-llama/Meta-Llama-3.1-405B-Instruct'
        elif key == 'Hyperbolic_LLM_Llama_3_1_405B_FP8': 
            return 'meta-llama/Meta-Llama-3.1-405B-FP8'
        elif key == 'Hyperbolic_LLM_Llama_3_1_8B': 
            return 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        elif key == 'Hyperbolic_LLM_Llama_3_1_70B': 
            return 'meta-llama/Meta-Llama-3.1-70B-Instruct'
        elif key == 'Hyperbolic_LLM_Llama_3_70B': 
            return 'meta-llama/Meta-Llama-3-70B-Instruct'
        # elif key == 'Hyperbolic_LLM_Hermes_3_70B': 
            # return 'NousResearch/Hermes-3-Llama-3.1-70B'
        elif key == 'Hyperbolic_LLM_DeepSeek_V2_5': 
            return 'deepseek-ai/DeepSeek-V2.5'
        

        ### Mistral LOCAL
        #LOCAL_MISTRAL_NEMO_INSTRUCT_2407  'LOCAL Mistral Nemo Instruct 2407 mistralai/Mistral-Nemo-Instruct-2407
        elif key == 'LOCAL_MISTRAL_NEMO_INSTRUCT_2407':
            return 'Mistral-Nemo-Instruct-2407'
        
        elif key == 'LOCAL_MIXTRAL_8X7B_INSTRUCT_V01':
            return 'Mixtral-8x7B-Instruct-v0.1'
        
        elif key == 'LOCAL_MISTRAL_7B_INSTRUCT_V02':
            return 'Mistral-7B-Instruct-v0.3'
        
        ### Mistral LOCAL CPU
        elif key == 'LOCAL_CPU_MISTRAL_7B_INSTRUCT_V02_GGUF':
            return 'Mistral-7B-Instruct-v0.2-GGUF'
        


        ### LOCAL custom fine-tuned
        elif key == 'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05':
            return 'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05'
        


        else:
            raise ValueError(f'Invalid model name {key}. See model_maps.py') 

    @classmethod
    def get_models_gui_list(cls):
        return cls.MODELS_LOCAL + cls.MODELS_GOOGLE + cls.MODELS_OPENAI + cls.MODELS_OPENAI_AZURE + cls.MODELS_MISTRAL + cls.MODELS_HYPERBOLIC
    
    @classmethod 
    def get_models_gui_list_family(cls, family=None):
        if family and family in cls.MODEL_FAMILY:
            return cls.MODEL_FAMILY[family]
        all_models = []
        for family_models in cls.MODEL_FAMILY.values():
            all_models.extend(family_models)
        return all_models

    @classmethod
    def get_version_mapping_cost(cls, key):
        return cls.version_mapping_cost.get(key, None)
    
    @classmethod
    def get_all_mapping_cost(cls):
        return cls.version_mapping_cost
    
