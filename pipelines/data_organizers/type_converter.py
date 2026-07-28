import logging

import pandas
import pyreadstat

from pipelines.data_organizers.file_pathways import NON_CSVS_FOLDER, UNMERGED_CSVS_FOLDER


def to_csv(convert_dir=NON_CSVS_FOLDER, output_path=UNMERGED_CSVS_FOLDER):
    logger = logging.getLogger(__name__)
    logger.info('Convert beginning')
    for raw_file in convert_dir.iterdir():
        file_type = raw_file.suffix
        file_name = raw_file.stem

        if file_type == '.sav':
            logger.info(f'SPSS file detected: {file_name}')
            df, meta = pyreadstat.read_sav(raw_file)
            output = output_path / f"{file_name}.csv"
            df.to_csv(output, index=False)
            logger.info(f'Saved: {output}')

        elif file_type == '.json':
            logger.info(f'JSON file detected: {file_name}')
            df = pandas.read_json(raw_file)
            output = output_path / f"{file_name}.csv"
            df.to_csv(output, index=False)
            logger.info(f'Saved: {output}')

        elif file_type in ('.xlsx', '.xls'):
            logger.info(f'Excel file detected: {file_name}')
            df = pandas.read_excel(raw_file)
            output = output_path / f"{file_name}.csv"
            df.to_csv(output, index=False)
            logger.info(f'Saved: {output}')

        elif file_type == '.xml':
            logger.info(f'XML file detected: {file_name}')
            df = pandas.read_xml(raw_file)
            output = output_path / f"{file_name}.csv"
            df.to_csv(output, index=False)
            logger.info(f'Saved: {output}')

        else:
            logger.info(f'Viable data type not detected: {file_name}')
