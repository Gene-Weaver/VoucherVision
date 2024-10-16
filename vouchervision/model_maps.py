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
        'GEMINI_PRO': '#1E00FF',  # 
        'GEMINI_1_5_FLASH': '#1E00FF',  # gemini-1.5-flash
        'GEMINI_1_5_PRO': '#1E00FF',  # gemini-1.5-pro
        

        'AZURE_GPT_4': '#800080',  # Purple
        # 'AZURE_GPT_4_TURBO_1106': '#9370DB',  # Medium Purple
        # 'AZURE_GPT_4_TURBO_0125': '#9370DB',  # Medium Purple
        # 'AZURE_GPT_4_32K': '#8A2BE2',  # Blue Violet
        # 'AZURE_GPT_3_5_INSTRUCT': '#9400D3',  # Dark Violet
        # 'AZURE_GPT_3_5': '#9932CC',  # Dark Orchid

        'OPEN_MISTRAL_7B': '#FFA07A',  # Light Salmon
        'OPEN_MIXTRAL_8X7B': '#FF8C00',  # Dark Orange
        'MISTRAL_SMALL': '#FF6347',  # Tomato
        'MISTRAL_MEDIUM': '#FF4500',  # Orange Red
        'MISTRAL_LARGE': '#800000',  # Maroon

        'LOCAL_MIXTRAL_8X7B_INSTRUCT_V01': '#000000',  # Black
        'LOCAL_MISTRAL_7B_INSTRUCT_V02': '#4a4a4a',  # Gray
        #  mistralai/Mistral-Nemo-Instruct-2407
        'LOCAL_MISTRAL_NEMO_INSTRUCT_2407': '#000000',  # Black

        'LOCAL_CPU_MISTRAL_7B_INSTRUCT_V02_GGUF': '#bababa',  # Gray

        'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05': '#bababa',  # Gray
    }

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
                    'Azure GPT 4',
                    #  'Azure GPT 4 32k',
                    #  'Azure GPT 4 Turbo 0125-preview',
                    #  'Azure GPT 4 Turbo 1106-preview',
                    #  'Azure GPT 3.5 Turbo',
                    #  'Azure GPT 3.5 Instruct',
                    ]
    
    MODELS_GOOGLE = [
                    # 'PaLM 2 text-bison@001',
                     'PaLM 2 text-bison@002',
                     'PaLM 2 text-unicorn@001',
                     'Gemini 1.0 Pro',
                     'Gemini 1.5 Flash',
                     'Gemini 1.5 Pro',
                     ]
    
    MODELS_MISTRAL = ['Mistral Small',
                      'Mistral Medium',
                      'Mistral Large',
                      'Open Mixtral 8x7B',
                      'Open Mistral 7B',
                      ]

    MODELS_LOCAL = ['LOCAL Mistral Nemo Instruct 2407', 
                    'LOCAL Mixtral 8x7B Instruct v0.1',
                    'LOCAL Mistral 7B Instruct v0.2',
                    'LOCAL CPU Mistral 7B Instruct v0.2 GGUF',
                    'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05']

    MODELS_GUI_DEFAULT = 'Gemini 1.5 Flash' #'Azure GPT 4' # 'GPT 4 Turbo 1106-preview'

    MODEL_FAMILY = {
                    'OpenAI': MODELS_OPENAI,
                    'OpenAI Azure': MODELS_OPENAI_AZURE,
                    'Google': MODELS_GOOGLE, 
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
        # 'Azure GPT 4 Turbo 0125-preview': 'AZURE_GPT_4_TURBO_0125',
        # 'Azure GPT 4 Turbo 1106-preview': 'AZURE_GPT_4_TURBO_1106',
        # 'Azure GPT 3.5 Instruct': 'AZURE_GPT_3_5_INSTRUCT',
        # 'Azure GPT 3.5 Turbo': 'AZURE_GPT_3_5',

        'Gemini 1.0 Pro': 'GEMINI_PRO',
        'Gemini 1.5 Flash': 'GEMINI_1_5_FLASH',  # gemini-1.5-flash
        'Gemini 1.5 Pro': 'GEMINI_1_5_PRO',  # gemini-1.5-pro

        'PaLM 2 text-unicorn@001': 'PALM2_TU_1',
        # 'PaLM 2 text-bison@001': 'PALM2_TB_1',
        'PaLM 2 text-bison@002': 'PALM2_TB_2',

        'Mistral Large': 'MISTRAL_LARGE',
        'Mistral Medium': 'MISTRAL_MEDIUM',
        'Mistral Small': 'MISTRAL_SMALL',
        'Open Mixtral 8x7B': 'OPEN_MIXTRAL_8X7B',
        'Open Mistral 7B': 'OPEN_MISTRAL_7B',

        'LOCAL Mistral Nemo Instruct 2407': 'LOCAL_MISTRAL_NEMO_INSTRUCT_2407', 
        'LOCAL Mixtral 8x7B Instruct v0.1': 'LOCAL_MIXTRAL_8X7B_INSTRUCT_V01',
        'LOCAL Mistral 7B Instruct v0.2': 'LOCAL_MISTRAL_7B_INSTRUCT_V02',

        'LOCAL CPU Mistral 7B Instruct v0.2 GGUF': 'LOCAL_CPU_MISTRAL_7B_INSTRUCT_V02_GGUF',

        'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05': 'phyloforfun/mistral-7b-instruct-v2-bnb-4bit__HLT_MICH_Angiospermae_SLTPvC_v1-0_medium_OCR-C25-L25-E50-R05',
    }

    @classmethod
    def get_version_has_key(cls, key, has_key_openai, has_key_azure_openai, has_key_google_application_credentials, has_key_mistral):
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
            # 'Azure GPT 4 Turbo 1106-preview': has_key_azure_openai,
            # 'Azure GPT 4 Turbo 0125-preview': has_key_azure_openai,
            # 'Azure GPT 4 32k': has_key_azure_openai,

            # 'PaLM 2 text-bison@001':  has_key_google_application_credentials,
            'PaLM 2 text-bison@002':  has_key_google_application_credentials,
            'PaLM 2 text-unicorn@001':  has_key_google_application_credentials,
            'Gemini 1.0 Pro':  has_key_google_application_credentials,
            'Gemini 1.5 Flash':  has_key_google_application_credentials,
            'Gemini 1.5 Pro':  has_key_google_application_credentials,

            'Mistral Small':  has_key_mistral,
            'Mistral Medium':  has_key_mistral,
            'Mistral Large':  has_key_mistral,
            'Open Mixtral 8x7B':  has_key_mistral,
            'Open Mistral 7B':  has_key_mistral,

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
            # 'Azure GPT 4 Turbo 1106-preview': True,
            # 'Azure GPT 4 Turbo 0125-preview': True,
            # 'Azure GPT 4 32k': True,

            # 'PaLM 2 text-bison@001':  False,
            'PaLM 2 text-bison@002':  False,
            'PaLM 2 text-unicorn@001':  False,
            'Gemini 1.0 Pro':  False,
            'Gemini 1.5 Flash':  False,
            'Gemini 1.5 Pro':  False,

            'Mistral Small':  False,
            'Mistral Medium':  False,
            'Mistral Large':  False,
            'Open Mixtral 8x7B':  False,
            'Open Mistral 7B':  False,

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
        
        elif key == 'GEMINI_PRO':
            return 'gemini-1.0-pro' 
        
        elif key == 'GEMINI_1_5_FLASH':
            return 'gemini-1.5-flash' 

        elif key == 'GEMINI_1_5_PRO':
            return 'gemini-1.5-pro' 
        
        ### Mistral 
        elif key == 'OPEN_MISTRAL_7B':
            return 'open-mistral-7b'
        
        elif key == 'OPEN_MIXTRAL_8X7B':
            return 'open-mixtral-8x7b'
        
        elif key == 'MISTRAL_SMALL':
            return 'mistral-small-latest'
        
        elif key == 'MISTRAL_MEDIUM':
            return 'mistral-medium-latest'
        
        elif key == 'MISTRAL_LARGE':
            return 'mistral-large-latest'
        

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
        return cls.MODELS_LOCAL + cls.MODELS_GOOGLE + cls.MODELS_OPENAI + cls.MODELS_OPENAI_AZURE + cls.MODELS_MISTRAL
    
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