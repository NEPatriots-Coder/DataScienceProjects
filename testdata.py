import pandas as pd
import numpy as np

# Generate mock data
np.random.seed(42)  # For reproducibility same random numbers on each run
num_samples = 100 # Number of samples to generate we can do less more samples if needed

# Define tolerance thresholds for each parameter
tolerances = {
    "Nominal_Diameter": (342.0, 344.0),  # Min, Max acceptable values
    "Bore_Diameter": (24.0, 26.0),
    "LTB": (28.0, 32.0),
    "Pitch": (0.615, 0.635),
    "Roller_Width": (0.365, 0.385),
    "Roller_Diameter": (390.0, 410.0)
}

mock_data = pd.DataFrame({
    "Nominal_Diameter": np.random.normal(loc=343, scale=2, size=num_samples),  # ~343 with variation
    "Bore_Diameter": np.random.normal(loc=25, scale=0.5, size=num_samples),   # Example bore size
    "LTB": np.random.normal(loc=30, scale=1, size=num_samples),              # Length Through Bore
    "Pitch": np.random.normal(loc=0.625, scale=0.01, size=num_samples),       # 5/8" = 0.625 inches
    "Roller_Width": np.random.normal(loc=0.375, scale=0.005, size=num_samples),  # 3/8" = 0.375 inches
    "Roller_Diameter": np.random.normal(loc=400, scale=5, size=num_samples)   # ~400 with variation
})

# Check each parameter against its tolerance and flag out-of-tolerance values
for param, (min_val, max_val) in tolerances.items():
    mock_data[f"{param}_InTolerance"] = (mock_data[param] >= min_val) & (mock_data[param] <= max_val)
    
# Add a column that indicates if all parameters are within tolerance
mock_data["All_InTolerance"] = mock_data[[f"{param}_InTolerance" for param in tolerances.keys()]].all(axis=1)

# Save to CSV for testing
mock_data.to_csv("results.csv", index=False)

# Display first few rows and summary of out-of-tolerance items
print(mock_data.head())
print("\nOut of tolerance summary:")
for param in tolerances.keys():
    out_count = (~mock_data[f"{param}_InTolerance"]).sum()
    print(f"{param}: {out_count} out of {num_samples} samples out of tolerance")
print(f"Total parts with any parameter out of tolerance: {(~mock_data['All_InTolerance']).sum()}")