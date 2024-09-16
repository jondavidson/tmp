import polars as pl
from typing import List, Dict, Optional
import numpy as np

class FeatureParameterMapper:
    def __init__(self, override_behavior: str = 'supersede'):
        """
        Initialize the FeatureParameterMapper.

        Parameters:
        - override_behavior: 'supersede' or 'additive'
            - 'supersede': Overrides replace base feature contributions.
            - 'additive': Overrides are added to base feature contributions.
        """
        self.parameter_labels = []
        self.parameter_mapping = {}  # Maps label to index
        self.offset = 0  # Current offset for parameter indices
        self.feature_offsets = {}  # Starting index for each feature
        self.df = None  # Placeholder for the DataFrame
        self.parameters = None  # Parameter vector
        self.overrides = []  # List of override definitions
        self.override_behavior = override_behavior  # Model behavior setting

    def fit(self, df: pl.DataFrame, features: List[str]):
        """
        Fit the mapper to the base additive features.
        """
        self.df = df.clone()  # Clone to avoid modifying original DataFrame
        for feature in features:
            self._add_feature(feature)
        # Initialize parameter vector
        self.parameters = np.zeros(self.offset)
        return self

    def _add_feature(self, feature: str):
        """
        Add a base feature to the parameter mapping.
        """
        # Convert to categorical with a consistent order
        self.df = self.df.with_column(pl.col(feature).cast(pl.Categorical))

        # Get unique categories
        unique_cats = self.df.select(pl.col(feature)).unique().sort(by=feature)
        n_unique = unique_cats.height
        cats = unique_cats.get_column(feature).to_list()

        # Store the starting offset for this feature
        start_offset = self.offset
        self.feature_offsets[feature] = start_offset

        # Create labels and update mappings
        for cat in cats:
            label = f"{feature}_{cat}"
            self.parameter_labels.append(label)
            self.parameter_mapping[label] = self.offset
            self.offset += 1

        # Adjust codes
        self.df = self.df.with_column(
            (pl.col(feature).cat.codes() + start_offset).alias(feature + '_idx')
        )

    def add_override(self, features: List[str], categories: Dict[str, str], priority: int = 0, name: Optional[str] = None):
        """
        Add an overriding term for specific category combinations with a priority level.
        """
        # Generate the override feature name
        if not name:
            name = "_".join([f"{k}_{v}" for k, v in categories.items()]) + "_override"

        # Create the override indicator column
        condition = pl.lit(True)
        for feature, category in categories.items():
            condition = condition & (pl.col(feature).cast(str) == pl.lit(category))

        indicator_col = name + '_indicator'
        self.df = self.df.with_column(
            condition.cast(pl.Int32).alias(indicator_col)
        )

        # Add the override parameter to the mapping
        self._add_override_parameter(name)

        # Record the override definition with priority
        self.overrides.append({
            'features': list(categories.keys()),
            'name': name,
            'categories': categories,
            'indicator_col': indicator_col,
            'priority': priority
        })
        # Sort overrides by priority (higher priority first)
        self.overrides.sort(key=lambda x: x['priority'], reverse=True)
        return self

    def _add_override_parameter(self, name: str):
        """
        Add the override parameter to the mapping.
        """
        label = name
        self.parameter_labels.append(label)
        self.parameter_mapping[label] = self.offset
        self.offset += 1
        # Initialize parameter vector (expand if necessary)
        if self.parameters is not None and len(self.parameters) < self.offset:
            self.parameters = np.append(self.parameters, 0)

    def get_parameter_vector_length(self):
        """
        Get the total number of parameters.
        """
        return self.offset

    def get_parameter_labels(self):
        """
        Get the list of parameter labels.
        """
        return self.parameter_labels

    def get_dataframe(self):
        """
        Get the DataFrame with index columns.
        """
        return self.df

    def predict(self, idx_row: int):
        """
        Compute prediction for a single data point, accounting for overrides.
        """
        prediction = 0.0
        applied_overrides = []

        # Check for active overrides
        for override in self.overrides:
            if self.df[override['indicator_col']][idx_row] == 1:
                idx = self.parameter_mapping[override['name']]
                prediction += self.parameters[idx]
                applied_overrides.append(override['name'])
                # If only one override should apply, uncomment the next line
                # break  # Stop after applying the highest priority override

        # Decide whether to include base features
        if self.override_behavior == 'supersede':
            if not applied_overrides:
                # Sum base feature parameters
                indices = []
                for feature in self.feature_offsets.keys():
                    idx = self.df[feature + '_idx'][idx_row]
                    indices.append(idx)
                prediction += np.sum(self.parameters[indices])
            else:
                # Overrides supersede base features; do not add base parameters
                pass
        elif self.override_behavior == 'additive':
            # Include base features regardless of overrides
            indices = []
            for feature in self.feature_offsets.keys():
                idx = self.df[feature + '_idx'][idx_row]
                indices.append(idx)
            prediction += np.sum(self.parameters[indices])
        else:
            raise ValueError("Invalid override_behavior. Choose 'supersede' or 'additive'.")

        return prediction

    def update_parameters(self, idx_row: int, target: float, learning_rate: float):
        """
        Update parameters for a single data point using SGD, accounting for overrides.
        """
        # Compute prediction
        prediction = self.predict(idx_row)
        # Compute error
        error = prediction - target

        # Check for active overrides
        applied_overrides = []
        for override in self.overrides:
            if self.df[override['indicator_col']][idx_row] == 1:
                # Update the override parameter
                idx = self.parameter_mapping[override['name']]
                self.parameters[idx] -= learning_rate * error
                applied_overrides.append(override['name'])
                # If only one override should apply, uncomment the next line
                # break  # Stop after updating the highest priority override

        # Decide whether to update base features
        if self.override_behavior == 'supersede':
            if not applied_overrides:
                # Update base feature parameters
                for feature in self.feature_offsets.keys():
                    idx = self.df[feature + '_idx'][idx_row]
                    self.parameters[idx] -= learning_rate * error
            else:
                # Overrides supersede base features; do not update base parameters
                pass
        elif self.override_behavior == 'additive':
            # Always update base feature parameters
            for feature in self.feature_offsets.keys():
                idx = self.df[feature + '_idx'][idx_row]
                self.parameters[idx] -= learning_rate * error
        else:
            raise ValueError("Invalid override_behavior. Choose 'supersede' or 'additive'.")

    def transform(self, df_new: pl.DataFrame):
        """
        Transform a new DataFrame using the fitted mappings.
        """
        # Implement transformation logic, ensuring that overrides are handled
        pass
