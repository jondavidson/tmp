#!/usr/bin/env python3

import polars as pl
import click
import re
import sys


def read_parquet_file(file_path):
    df = pl.read_parquet(file_path)
    return df


def search_dataframe(file_path, pattern, columns=None, case_sensitive=False):
    regex_flag = '(?i)' if not case_sensitive else ''
    regex_pattern = f"{regex_flag}{pattern}"

    if columns is None:
        df = pl.scan_parquet(file_path)
        # Get string columns
        str_columns = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
        df = df.select(str_columns)
    else:
        str_columns = columns
        df = pl.scan_parquet(file_path).select(str_columns)

    # Build the filter expression
    combined_filter = None
    for col in str_columns:
        filter_expr = pl.col(col).str.contains(regex_pattern, literal=False, regex=True)
        if combined_filter is None:
            combined_filter = filter_expr
        else:
            combined_filter = combined_filter | filter_expr

    if combined_filter is not None:
        result_df = df.filter(combined_filter).collect(streaming=True)
        return result_df
    else:
        return pl.DataFrame()


def calculate_column_widths(df):
    """Calculate the maximum width for each column."""
    widths = {col: max(len(col), df[col].cast(str).str.lengths().max().item()) for col in df.columns}
    return widths


def format_fixed_width(df, widths, pattern, case_sensitive):
    """Format the DataFrame rows as fixed-width columns with colored matches."""
    header = "  ".join([col.ljust(widths[col]) for col in df.columns])
    rows = []
    for row in df.iter_rows():
        # Format and colorize each value
        row_str = "  ".join(
            colorize_match(str(value).ljust(widths[col]), pattern, case_sensitive)
            for value, col in zip(row, df.columns)
        )
        rows.append(row_str)
    return header, rows


def colorize_match(text, pattern, case_sensitive):
    """Color the matched pattern in the given text."""
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.sub(pattern, lambda m: f'\033[31m{m.group(0)}\033[0m', text, flags=flags)


def process_with_vectorization(df, pattern, case_sensitive):
    def colorize_column(col):
        return col.apply(lambda val: colorize_match(str(val), pattern, case_sensitive))

    # Apply colorization vectorized on each column
    colored_df = df.select([colorize_column(pl.col(col)).alias(col) for col in df.columns])

    # Write DataFrame directly as CSV to stdout
    colored_df.write_csv(sys.stdout, separator='\t')


@click.command()
@click.argument('pattern')
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.option('--columns', '-c', multiple=True, help='Columns to search in.')
@click.option('--case-sensitive', is_flag=True, help='Case sensitive search.')
@click.option('--fixed-width', is_flag=True, help='Output in fixed-width aligned columns.')
@click.option('--color', is_flag=True, help='Highlight matches in color.')
def main(pattern, files, columns, case_sensitive, fixed_width, color):
    for file_path in files:
        result_df = search_dataframe(
            file_path,
            pattern,
            columns=columns if columns else None,
            case_sensitive=case_sensitive
        )

        if not result_df.is_empty():
            # Add the file name to each row
            result_df = result_df.with_columns(pl.lit(file_path).alias('file'))
            # Place 'file' as the first column
            cols = ['file'] + [col for col in result_df.columns if col != 'file']
            result_df = result_df.select(cols)

            if fixed_width:
                # Calculate column widths for fixed-width formatting
                widths = calculate_column_widths(result_df)
                # Format the header and rows
                header, rows = format_fixed_width(result_df, widths, pattern, case_sensitive)
                # Print the formatted output
                print(header)
                for row in rows:
                    print(row)
            else:
                # Process with vectorization for color and CSV output
                if color:
                    process_with_vectorization(result_df, pattern, case_sensitive)
                else:
                    # Write DataFrame without color directly to stdout as CSV/tab-separated
                    result_df.write_csv(sys.stdout, separator='\t')
        else:
            continue  # No matches in this file


if __name__ == '__main__':
    main()
