import os
import glob
import shutil
from tqdm import tqdm
        

class FinetuneModel:
    def __init__(self, model_name: str, base_model: str, data_dir: str, tesstrain_dir: str) -> None:
        self.model_name = model_name
        self.base_model = base_model
        self.data_dir = data_dir
        self.tesstrain_dir = os.path.abspath(tesstrain_dir)


    def check_makefile(self) -> None:
        print("""
Antes de iniciar o finetune verifique se o arquivo Makefile contém os seguinte trecho de código:

$(PROTO_MODEL): $(OUTPUT_DIR)/unicharset $(TESSERACT_LANGDATA)
    if [ "$(OS)" = "Windows_NT" ]; then \\
            dos2unix "$(NUMBERS_FILE)"; \\
            dos2unix "$(PUNC_FILE)"; \\
            dos2unix "$(WORDLIST_FILE)"; \\
            dos2unix "$(LANGDATA_DIR)/$(MODEL_NAME)/$(MODEL_NAME).config"; \\
    fi
    combine_lang_model \\
      --input_unicharset $(OUTPUT_DIR)/unicharset \\
      --script_dir $(LANGDATA_DIR) \\
      --numbers $(NUMBERS_FILE) \\
      --puncs $(PUNC_FILE) \\
      --words $(WORDLIST_FILE) \\
      --output_dir $(DATA_DIR) \\
      $(RECODER) \\
      --lang $(MODEL_NAME)

Se sim subistitua-o por esse trecho:

$(PROTO_MODEL): $(OUTPUT_DIR)/unicharset $(TESSERACT_LANGDATA)
    combine_lang_model \\
      --input_unicharset $(OUTPUT_DIR)/unicharset \\
      --script_dir $(LANGDATA_DIR) \\
      --output_dir $(DATA_DIR) \\
      $(RECODER) \\
      --lang $(MODEL_NAME)
      
        """)


    def edit_makefile(self) -> None:
        cd_number, cd_punc, cd_word = None, None, None
        if self.tesstrain_dir == os.getcwd():
            makefile_path = 'Makefile'
        else:
            makefile_path = f'{self.tesstrain_dir}/Makefile'
        
        with open(makefile_path, 'r') as file:
            data = file.readlines()

        for i, line in enumerate(data):
            if '$(PROTO_MODEL): $(OUTPUT_DIR)/unicharset $(TESSERACT_LANGDATA)' in line:
                start_if = i+1
            if "combine_lang_model \\" in line:
                end_if = i
            if "--numbers $(NUMBERS_FILE) \\" in line:
                cd_number = i
            if "--puncs $(PUNC_FILE) \\" in line:
                cd_punc = i
            if "--words $(WORDLIST_FILE) \\" in line:
                cd_word = i

        if (cd_number is not None) and (cd_punc is not None) and (cd_word is not None):
            print("Editando Makefile...")
            
            lines_rm = [i for i in range(start_if, end_if)]
            lines_rm.extend([cd_number, cd_punc, cd_word])
            new_data = [e for i, e in enumerate(data) if i not in lines_rm]

            with open(makefile_path, 'w') as file:
                file.writelines(new_data)

            print("Makefile editado com sucesso.")
        else:
            print("Makefile já foi editado.")


    def prepare_tesstrain(self) -> None:
        clone = f'git clone https://github.com/tesseract-ocr/tesstrain.git {self.tesstrain_dir}'
        make_langdata = 'make tesseract-langdata'
        data_dir_abs = os.path.abspath(self.data_dir)
        
        if not os.path.exists(self.tesstrain_dir):
            os.system(clone)
        else:
            print('O diretório tesstrain já existe. Pulando para o próximo passo...')

        print(f'Mudando o diretório atual para {self.tesstrain_dir}')
        os.chdir(self.tesstrain_dir)

        print('Criando langdata...')
        if not os.path.exists(r'./data/langdata'):
            os.system(make_langdata)
        else:
            print('O diretório /data/langdata já existe. Pulando para o próximo passo...')

        print('Criando tessdata...')
        if not os.path.exists('./usr/share/tessdata'):
            os.makedirs('./usr/share/tessdata')
        if not os.path.exists(f'./usr/share/tessdata/{self.base_model}.traineddata'):
            os.system(f'wget -P usr/share/tessdata https://github.com/tesseract-ocr/tessdata_best/raw/main/{self.base_model}.traineddata')

        print(f'Criando {self.model_name}-ground-truth...')
        if os.path.exists(data_dir_abs):
            shutil.move(data_dir_abs, f'./data/{self.model_name}-ground-truth')

        # self.check_makefile()
        self.edit_makefile()

        if os.name == 'nt':
            print("""
            Caso esteja usando o Windows instale o Git e rode o seguinte comando antes de iniciar o finetune:
            set PATH=C:\\Program Files\\Git\\usr\\bin;%PATH%
            """)
        

    def run(self) -> None:
        make_training = f'make training MODEL_NAME={self.model_name} START_MODEL={self.base_model} TESSDATA=usr/share/tessdata FINETUNE_TYPE=Impact'

        os.chdir(self.tesstrain_dir)
        
        if os.name == 'nt':
            os.system("set PATH=C:\\Program Files\\Git\\usr\\bin;%PATH%")

        print('Iniciando treinamento do modelo...')
        os.system(make_training)

