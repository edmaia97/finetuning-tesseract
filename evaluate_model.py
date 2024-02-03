import os
import glob
import jiwer
import pytesseract
import pandas as pd
from tqdm import tqdm
        

class EvaluateModel:
    def __init__(self, model_name: str, data_dir: str) -> None:
        self.model_name = model_name
        self.data_dir = data_dir
        self.accepted_filetypes = ['jpg', 'png', 'tif']
        self.base_image_ext = 'jpg'


    def get_accepted_files(self) -> list[str]:
        accepted_files = []
        
        for ext in self.accepted_filetypes:
            accepted_files.extend(glob.glob(f"{self.data_dir}/*.{ext}"))
            
        return accepted_files


    def generate_evaluation_data(self, psm=6) -> None:
        images_paths = self.get_accepted_files()

        for img_path in tqdm(images_paths, desc='Gerando dados para avaliação'):
            eval_file = f'{self.data_dir}/{base_filename}.{self.model_name}.txt'
            if os.path.exists(eval_file):
                continue
            
            ocr_text = pytesseract.image_to_string(img_path, lang=self.model_name, config=f'--psm {psm}')
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            with open(eval_file, 'w', encoding='utf-8') as ocr_file:
                ocr_file.write(ocr_text)

    
    def get_data(self):
        images_paths = self.get_accepted_files()

        references = []
        ocrs = []
        for img in images_paths:
            with open(img.replace(self.base_image_ext, 'gt.txt'), 'r', encoding='utf-8') as f1, open(img.replace(self.base_image_ext, f'{self.model_name}.txt'), 'r', encoding='utf-8') as f2:
                output_gt = f1.readlines()
                output_ocr = f2.readlines()

            ref = ''.join(output_gt)
            if not ref.split():
                print(
                    f"\033[31m[Erro] Arquivos de referência (.gt.txt) não podem estar vazios.\n"
                    f"Pulando o arquivo '{os.path.abspath(img)}'...\033[0m"
                )
                continue
            references.append(ref)
            ocrs.append(''.join(output_ocr))

        return references, ocrs


    def get_data_dict(self):
        images_paths = self.get_accepted_files()

        data_list = []
        for img in images_paths:
            with open(img.replace(self.base_image_ext, 'gt.txt'), 'r', encoding='utf-8') as f1, open(img.replace(self.base_image_ext, f'{self.model_name}.txt'), 'r', encoding='utf-8') as f2:
                output_gt = f1.readlines()
                output_ocr = f2.readlines()

            ref = ''.join(output_gt)
            if not ref.split():
                print(
                    f"\033[31m[Erro] Arquivos de referência (.gt.txt) não podem estar vazios.\n"
                    f"Pulando o arquivo '{os.path.abspath(img)}'...\033[0m"
                )
                continue
            
            data_dict = {
                'original_img': img.split('_crop_')[0],
                'img_crop': img,
                'ref_text': ''.join(output_gt),
                'ocr_text': ''.join(output_ocr),
            }
            data_list.append(data_dict)
        
        return data_list


    def get_char_level_eval(self, overall=True):
        if overall:
            ref_texts, ocr_texts = self.get_data()
            output = jiwer.process_characters(ref_texts, ocr_texts)
            cer = round(output.cer * 100, 2)
            print(
                f"Métricas a Nível de Caracteres do modelo de OCR '{self.model_name}'\n"
                f"Character Error Rate(CER): {cer}%"
            )
        else:
            data_list = self.get_data_dict()
            df_output = pd.DataFrame.from_dict(data_list)
            df_output['cer'] = ''
            df_output['ref_text'] = df_output['ref_text'].apply(lambda x: x.replace('\n',''))
            df_output['ocr_text'] = df_output['ocr_text'].apply(lambda x: x.replace('\n',''))
            for index, row in df_output.iterrows():
                filename = row['img_crop']
                ref = row['ref_text']
                ocr = row['ocr_text']
                cer = jiwer.cer(ref, ocr)
                df_output.loc[df_output['img_crop'] == filename, 'cer'] = round(cer, 4)
            df_met = df_output[['original_img', 'cer']]
            df_met = df_met.groupby(['original_img']).mean()
            print(df_met)


    def get_word_level_eval(self, overall=True):
        if overall:
            ref_texts, ocr_texts = self.get_data()
            output = jiwer.process_words(ref_texts, ocr_texts)
            wer = round(output.wer * 100, 2)
            mer = round(output.mer * 100, 2)
            wil = round(output.wil * 100, 2)
            wip = round(output.wip * 100, 2)
            print(
                f"Métricas a Nível de Palavras do modelo de OCR '{self.model_name}'\n"
                f"Word Error Rate(WER): {wer}%\n"
                f"Match Error Rate(MER): {mer}%\n"
                f"Word Information Lost(WIL): {wil}%\n"
                f"Word Information Preserved(WIP): {wip}%\n"
            )
        else:
            data_list = self.get_data_dict()
            df_output = pd.DataFrame.from_dict(data_list)
            df_output['wer'] = ''
            df_output['mer'] = ''
            df_output['wil'] = ''
            df_output['wip'] = ''
            df_output['ref_text'] = df_output['ref_text'].apply(lambda x: x.replace('\n',''))
            df_output['ocr_text'] = df_output['ocr_text'].apply(lambda x: x.replace('\n',''))
            for index, row in df_output.iterrows():
                filename = row['img_crop']
                ref = row['ref_text']
                ocr = row['ocr_text']
                wer = jiwer.wer(ref, ocr)
                mer = jiwer.mer(ref, ocr)
                wil = jiwer.wil(ref, ocr)
                wip = jiwer.wip(ref, ocr)
                df_output.loc[df_output['img_crop'] == filename, 'wer'] = round(wer, 4)
                df_output.loc[df_output['img_crop'] == filename, 'mer'] = round(mer, 4)
                df_output.loc[df_output['img_crop'] == filename, 'wil'] = round(wil, 4)
                df_output.loc[df_output['img_crop'] == filename, 'wip'] = round(wip, 4)
            df_met = df_output[['original_img', 'wer', 'mer', 'wil', 'wip']]
            df_met = df_met.groupby(['original_img']).mean()
            print(df_met)

        
