# script2.py

import argparse
import pandas as pd
import os

def main(args):
    parser = argparse.ArgumentParser(description='Process dataset1 to create dataset2.')
    parser.add_argument('--input', type=str, required=True, help='Input dataset path.')
    parser.add_argument('--output', type=str, required=True, help='Output dataset path.')
    parser.add_argument('--multiplier', type=int, default=2, help='Multiplier value.')
    parser.add_argument('--start_date', type=str, required=False, help='Start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=False, help='End date (YYYY-MM-DD).')

    parsed_args = parser.parse_args(args)

    # Read input dataset
    df = pd.read_csv(parsed_args.input)

    # Process data
    df['value'] = df['value'] * parsed_args.multiplier

    # Ensure output directory exists
    os.makedirs(os.path.dirname(parsed_args.output), exist_ok=True)

    # Save output dataset
    df.to_csv(parsed_args.output, index=False)
    print(f"Dataset2 generated at {parsed_args.output}")

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
