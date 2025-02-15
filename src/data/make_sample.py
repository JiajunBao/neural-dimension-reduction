# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import logging
from pathlib import Path
from src.data.utils import import_raw_data, export_processed_data


def downsample_data(df, num_rows, seed):
    sampled_train_df = df.sample(n=num_rows, random_state=seed)
    sampled_test_df = df.sample(n=num_rows // 3, random_state=(seed + 10))
    return sampled_train_df, sampled_test_df


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    # get arguments
    parser = ArgumentParser(description='Arguments for dataset processing')
    parser.add_argument('--input_path', type=Path, required=True, default=None,
                        help='the input path to the input data')
    parser.add_argument('--output_dir', type=Path, required=True, default=None,
                        help='the output directory to save sampled data')
    parser.add_argument('--num_rows', type=int, required=True, default=1000,
                        help='the number of rows to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='the random seed of the whole process')
    args = parser.parse_args()

    logger.info(f'reading data from {args.input_path}')
    df = import_raw_data(args.input_path)
    sampled_train_df, sampled_test_df = downsample_data(df, args.num_rows, args.seed)
    export_processed_data(sampled_train_df, args.output_dir / 'sample' / 'train.csv')
    export_processed_data(sampled_test_df, args.output_dir / 'sample' / 'dev.csv')
    logger.info(f'saved data at {args.output_dir / "sample"}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
