# script3.py

import argparse
import pandas as pd
import os

def main(args):
    parser = argparse.ArgumentParser(description='Combine dataset1 and dataset2 to create dataset3.')
    parser.add_argument('--input1', type=str, required=True, help='Input dataset1 path.')
    parser.add_argument('--input2', type=str, required=True, help='Input dataset2 path.')
    parser.add_argument('--output', type=str, required=True, help='Output dataset path.')
    parser.add_argument('--start_date', type=str, required=False, help='Start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=False, help='End date (YYYY-MM-DD).')

    parsed_args = parser.parse_args(args)

    # Read input datasets
    df1 = pd.read_csv(parsed_args.input1)
    df2 = pd.read_csv(parsed_args.input2)

    # Merge datasets
    df_combined = pd.merge(df1, df2, on='date', suffixes=('_dataset1', '_dataset2'))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(parsed_args.output), exist_ok=True)

    # Save combined dataset
    df_combined.to_csv(parsed_args.output, index=False)
    print(f"Dataset3 generated at {parsed_args.output}")

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
