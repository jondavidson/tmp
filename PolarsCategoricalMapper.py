import polars as pl
import json
from typing import List, Dict, Any

class PolarsCategoricalMapper:
    """
    A class to convert string columns in a Polars DataFrame to categorical types
    and create a global mapping of all categories to unique indices in a single vector.
    """

    def __init__(self, parquet_path: str):
        """
        Initialize the mapper with the path to the Parquet file.
        
        Parameters:
            parquet_path (str): Path to the Parquet file.
        """
        self.parquet_path = parquet_path
        self.lazy_df = pl.scan_parquet(parquet_path)
        self.schema = self.lazy_df.schema
        self.string_cols = [col for col, dtype in self.schema.items() if dtype == pl.Utf8]
        self.category_mappings: Dict[str, Dict[str, int]] = {}
        self.global_mapping: Dict[str, int] = {}
        self.vector_length: int = 0
        self.processed_df: pl.DataFrame = pl.DataFrame()

    def convert_to_categorical(self):
        """
        Convert identified string columns to categorical types using lazy evaluation.
        """
        self.lazy_df = self.lazy_df.with_columns(
            [pl.col(col).cast(pl.Categorical).alias(col) for col in self.string_cols]
        )
        return self

    def collect_data(self):
        """
        Execute the lazy operations and collect the DataFrame.
        
        Returns:
            pl.DataFrame: The processed DataFrame with categorical columns.
        """
        self.processed_df = self.lazy_df.collect()
        return self.processed_df

    def create_category_mappings(self):
        """
        Create individual and global category mappings.
        """
        current_index = 0
        for col in self.string_cols:
            categorical_series = self.processed_df[col].to_physical()
            categories = categorical_series.categories
            mapping = {category: idx + current_index for idx, category in enumerate(categories)}
            self.category_mappings[col] = mapping
            current_index += len(categories)
        
        self.vector_length = current_index
        # Combine all mappings into a single global mapping
        for col_mapping in self.category_mappings.values():
            self.global_mapping.update(col_mapping)
        
        return self

    def save_mappings(self, filepath: str):
        """
        Save the category mappings to a JSON file.
        
        Parameters:
            filepath (str): Path to the JSON file where mappings will be saved.
        """
        with open(filepath, 'w') as f:
            json.dump({
                'category_mappings': self.category_mappings,
                'global_mapping': self.global_mapping,
                'vector_length': self.vector_length
            }, f, indent=4)
    
    def load_mappings(self, filepath: str):
        """
        Load the category mappings from a JSON file.
        
        Parameters:
            filepath (str): Path to the JSON file from which mappings will be loaded.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.category_mappings = data['category_mappings']
            self.global_mapping = data['global_mapping']
            self.vector_length = data['vector_length']
    
    def encode_row(self, row: Dict[str, Any]) -> List[int]:
        """
        Encode a single row's categorical features into a list of indices.
        
        Parameters:
            row (Dict[str, Any]): A dictionary representing a row with string categorical values.
        
        Returns:
            List[int]: A list of unique indices corresponding to the categories in the row.
        """
        indices = []
        for col in self.string_cols:
            category = row.get(col)
            if category is not None:
                index = self.category_mappings[col].get(category)
                if index is not None:
                    indices.append(index)
        return indices

    def encode_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Encode all rows in a DataFrame's categorical columns into lists of indices.
        
        Parameters:
            df (pl.DataFrame): The DataFrame to encode.
        
        Returns:
            pl.DataFrame: A new DataFrame with an additional 'encoded' column containing the index lists.
        """
        # Create a list of expressions that map each string column to its global index
        mapping_expressions = [
            pl.col(col).map_dict(self.global_mapping).alias(f"{col}_idx")
            for col in self.string_cols
        ]
        
        # Apply the mapping expressions to get index columns
        indexed_df = df.with_columns(mapping_expressions)
        
        # Collect the index columns into a list per row
        encoded_df = indexed_df.with_columns(
            pl.concat_list([f"{col}_idx" for col in self.string_cols]).alias("encoded")
        )
        
        # Optionally, drop the intermediate index columns
        encoded_df = encoded_df.drop([f"{col}_idx" for col in self.string_cols])
        
        return encoded_df

    def get_vector_length(self) -> int:
        """
        Get the total length of the global vector.
        
        Returns:
            int: The length of the vector.
        """
        return self.vector_length

    def get_global_mapping(self) -> Dict[str, int]:
        """
        Get the global category to index mapping.
        
        Returns:
            Dict[str, int]: The global mapping.
        """
        return self.global_mapping
