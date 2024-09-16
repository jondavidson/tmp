import numpy as np
import polars as pl

# Sample DataFrame
df = pl.DataFrame({
    'feature1': ['A', 'B', 'C', 'A', 'B'],
    'feature2': ['X', 'Y', 'Z', 'Y', 'X'],
    'feature3': ['foo', 'bar', 'baz', 'foo', 'bar'],
    'target': [10, 20, 30, 15, 25]  # Sample target values
})

# Define base features
base_features = ['feature1', 'feature2', 'feature3']

# Define overrides
overrides = [
    {
        'features': ['feature1', 'feature2'],
        'categories': {'feature1': 'A', 'feature2': 'X'},
        'priority': 2,
        'name': 'override_A_X'
    },
    {
        'features': ['feature1'],
        'categories': {'feature1': 'A'},
        'priority': 1,
        'name': 'override_A'
    }
]

# Function to run the model with specified override behavior
def run_model(override_behavior):
    print(f"\n--- Running model with override_behavior='{override_behavior}' ---")
    # Initialize the mapper
    mapper = FeatureParameterMapper(override_behavior=override_behavior)
    
    # Fit the mapper with base features
    mapper.fit(df, base_features)
    
    # Add overrides
    for override in overrides:
        mapper.add_override(
            features=override['features'],
            categories=override['categories'],
            priority=override['priority'],
            name=override['name']
        )
    
    # Initialize parameters
    np.random.seed(0)  # For reproducibility
    mapper.parameters = np.random.randn(mapper.get_parameter_vector_length())
    
    # Training loop
    learning_rate = 0.01
    for idx_row in range(len(df)):
        target = df['target'][idx_row]
        mapper.update_parameters(idx_row, target, learning_rate)
    
    # Retrieve parameter labels and values
    parameter_labels = mapper.get_parameter_labels()
    parameter_values = mapper.parameters
    
    # Display parameter values
    print("\nParameter labels and values:")
    for label, value in zip(parameter_labels, parameter_values):
        print(f"{label}: {value:.4f}")
    
    # Predictions
    predictions = []
    for idx_row in range(len(df)):
        prediction = mapper.predict(idx_row)
        predictions.append(prediction)
    
    df_result = df.with_column(pl.Series('prediction', predictions))
    
    print("\nDataFrame with predictions:")
    print(df_result)

# Run model with 'supersede' behavior
run_model('supersede')

# Run model with 'additive' behavior
run_model('additive')
