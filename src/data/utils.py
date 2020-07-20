import pandas as pd
from pathlib import Path


def import_raw_data(input_path):
    df = pd.read_csv(input_path, header=None)
    return df


def export_processed_data(processed_df: pd.DataFrame, output_path: Path):
    output_path.parents[0].mkdir(exist_ok=True, parents=True)
    processed_df.to_csv(output_path, header=False, index=False)
    return True
