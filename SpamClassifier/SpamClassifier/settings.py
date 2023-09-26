import os

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw_data')
NORMALIZED_DATA_PATH = os.path.join(DATA_PATH, 'normalized_data')
SEPARATED_DATA_PATH = os.path.join(RAW_DATA_PATH, "separated_files")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed_data')

folder_path = RAW_DATA_PATH
paths_list = os.listdir(folder_path)