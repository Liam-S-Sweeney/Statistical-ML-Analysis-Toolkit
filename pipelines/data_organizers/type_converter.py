import sys
import csv
import json
import os
import pyreadstat
import pandas as pd
import re
from pipelines.data_organizers.file_pathways import NON_CSVS_FOLDER, UNMERGED_CSVS_FOLDER

def to_csv(convert_dir=NON_CSVS_FOLDER, output_path=UNMERGED_CSVS_FOLDER):
    print('Convert beginning')
    for raw_file in convert_dir.iterdir():
        with open(raw_file, 'r') as rf:
            file_type = os.path.splitext(raw_file)[1]

            raw_file_name = os.path.splitext(raw_file)[0]
            file_name = re.split(r'\\', raw_file_name)[1]

            if file_type == '.sav':
                print('SPSS file Detected')
                df, meta = pyreadstat.read_sav(raw_file)
                output = os.path.join(output_path, file_name) + '.csv'
                df.to_csv(output, index=False)

            elif file_type == '.json':
                print('JSON file Detected')

            elif file_type == '.xlsx' or file_type == 'xls':
                print('EXCEL file Detected')

            elif file_type == '.xml':
                print('XML file Detected')

            elif file_type == '.tsv':
                print('TSV file Detected')

            else:
                print('Viable data type not detected:', file_name)

to_csv()