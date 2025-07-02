import pandas as pd
import os
from .config_loader import load_output_path_config


def save_csv(data_dict, file_path):

    rows_dict = [{'Vehicle Category': k, 'Issued Date': v[0], 'Expiry Date': v[1]} for k, v in data_dict.items()]

    df = pd.DataFrame(rows_dict)

    file_name = os.path.basename(file_path).split('.')[0]

    output_path = load_output_path_config()

    df.to_csv(output_path+file_name+'.csv')