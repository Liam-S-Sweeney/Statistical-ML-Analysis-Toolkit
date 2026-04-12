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
        file_type = raw_file.suffix
        file_name = raw_file.stem

        if file_type == '.sav':
            print(f'SPSS file detected: {file_name}')
            df, meta = pyreadstat.read_sav(raw_file)
            output = output_path / f"{file_name}.csv"
            df.to_csv(output, index=False)
            print(f'Saved: {output}')

        elif file_type == '.json':
            print('JSON file detected')

        elif file_type in ('.xlsx', '.xls'):
            print('Excel file detected')

        elif file_type == '.xml':
            print('XML file detected')

        elif file_type == '.tsv':
            print('TSV file detected')

        else:
            print(f'Viable data type not detected: {file_name}')