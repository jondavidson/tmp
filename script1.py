# script1.py

import argparse
import pandas as pd
import os

def main(args):
    parser = argparse.ArgumentParser(description='Generate dataset1.')
    parser.add_argument('--output', type=str, required=True, help='Output dataset path.')
    parser.add_argument('--start_date', type=str, required=False, help='Start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, required=False, help='End date (YYYY-MM-DD).')

    parsed_args = parser.parse_args(args)

    # Generate sample data
    data = {
        'date': pd.date_range(start=parsed_args.start_date, end=parsed_args.end_date),
        'value': range(1, (pd.to_datetime(parsed_args.end_date) - pd.to_datetime(parsed_args.start_date)).days + 2)
    }
    df = pd.DataFrame(data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(parsed_args.output), exist_ok=True)

    # Save dataset
    df.to_csv(parsed_args.output, index=False)
    print(f"Dataset1 generated at {parsed_args.output}")

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
