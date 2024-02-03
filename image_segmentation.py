import os
import glob
import cv2
import pytesseract
from paddleocr import PaddleOCR
from tqdm import tqdm
        

class ImageSegmentation:
    def __init__(self, images_dir: str, segments_dir: str) -> None:
        self.images_dir = images_dir
        self.segments_dir = segments_dir
        self.accepted_filetypes = ['jpg', 'png', 'tif']


    def get_accepted_files(self, files_dir: str) -> list[str]:
        accepted_files = []
        
        for ext in self.accepted_filetypes:
            accepted_files.extend(glob.glob(f"{files_dir}/*.{ext}"))
            
        return accepted_files


    def rename_crop(self, old_name, new_name):
        try:
            os.rename(old_name, new_name)
        except FileExistsError:
            os.remove(new_name)
            os.rename(old_name, new_name)


    def convert_crop(self, crop_name):
        try:
            img = cv2.imread(crop_name)
            cv2.imwrite(crop_name[:-3] + 'png', img)
            os.remove(crop_name)
        except:
            print('Cannot convert %s' % crop_name)


    def extract_segments(self) -> None:
        filenames = self.get_accepted_files(self.images_dir)

        for img_path in tqdm(filenames, desc='Extraindo segmentos'):
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            
            ocr = PaddleOCR(use_angle_cls=False, lang='pt', rec=False, show_log = False, save_crop_res=True, crop_res_save_dir=self.segments_dir)
            result = ocr.ocr(img_path, cls=False)
            
            for crop in glob.glob(self.segments_dir + "/mg_crop_*"):
                new_crop_name = crop.replace('mg_crop', f"{base_filename}_crop")
                self.rename_crop(crop, new_crop_name)
                self.convert_crop(new_crop_name)
                

    def generate_gt_files(self, psm = 7) -> None:
        segments = self.get_accepted_files(self.segments_dir)
        
        for segment in tqdm(segments, desc='Gerando arquivos GT'):
            base_filename = os.path.splitext(os.path.basename(segment))[0]
            
            gt_file = f'{self.segments_dir}/{base_filename}.gt.txt'
            if os.path.exists(gt_file):
                continue
            
            segment_ocr = pytesseract.image_to_string(segment, lang='por', config=f'--psm {psm}')
            with open(gt_file, 'w', encoding='utf-8') as ocr_file:
                if segment_ocr:
                    ocr_file.write(segment_ocr)
                else:
                    ocr_file.write('.')


    def run(self) -> None:
        self.extract_segments()
        self.generate_gt_files()

        
