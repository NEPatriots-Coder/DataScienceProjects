import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

print("Starting the Biden Activities Dashboard Generator...")

# --- Step 1: Load Data from Excel File ---
file_path = r'C:\Users\lamarw\Desktop\ComputerScience\DataScienceProjects\Biden_Locations.xlsx'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"ERROR: The file was not found at: {file_path}")
    exit()

try:
    print(f"\nReading data from Excel file: {file_path}")
    df = pd.read_excel(file_path)
    print(f"Successfully loaded {len(df)} rows from the file.")
except Exception as e:
    print(f"ERROR: Failed to read Excel file: {e}")
    exit()

# --- Step 2: Process and Analyze the Data ---
# Convert date column to datetime if it's not already
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.strftime('%Y-%m')
    df['year'] = df['date'].dt.year

# --- Step 3: Generate CSV Summary ---
output_csv = 'biden_activities_summary.csv'
print(f"\nGenerating CSV summary to {output_csv}...")
df.to_csv(output_csv, index=False)

# --- Step 4: Create a Simple HTML Dashboard ---
print("\nGenerating HTML dashboard...")

# Count activities by location
location_counts = df['location2'].value_counts().reset_index()
location_counts.columns = ['Location', 'Count']

# Count activities by month if date column exists
monthly_activity = df.groupby('month').size().reset_index() if 'month' in df.columns else None
if monthly_activity is not None:
    monthly_activity.columns = ['Month', 'Activities']

# Generate HTML content
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Biden Activities Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #0d3b66; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #0d3b66; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Biden Activities Dashboard</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total activities recorded: {len(df)}</p>
        <p>Date range: {df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns and not pd.isna(df['date'].min()) else 'N/A'} to 
                      {df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns and not pd.isna(df['date'].max()) else 'N/A'}</p>
        <p>Unique locations: {df['location2'].nunique() if 'location2' in df.columns else 'N/A'}</p>
        <p>Activities in Washington DC: {df['in_washington_dc'].sum() if 'in_washington_dc' in df.columns else 'N/A'}</p>
    </div>
    
    <h2>Activities by Location</h2>
    <table>
        <tr>
            <th>Location</th>
            <th>Number of Activities</th>
        </tr>
        {''.join(f"<tr><td>{row['Location']}</td><td>{row['Count']}</td></tr>" for _, row in location_counts.iterrows())}
    </table>
    
    <h2>Recent Activities</h2>
    <table>
        <tr>
            <th>Date</th>
            <th>Location</th>
            <th>Summary</th>
        </tr>
        {''.join(f"<tr><td>{row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'}</td><td>{row['location2']}</td><td>{row['location2_summary']}</td></tr>" for _, row in df.sort_values('date', ascending=False).head(10).iterrows())}
    </table>
    
    <p><a href="{output_csv}">Download full data as CSV</a></p>
</body>
</html>
"""

# Save the HTML dashboard
output_html = 'biden_activities_dashboard.html'
with open(output_html, 'w') as f:
    f.write(html_content)

print(f"\n🎉 Success! Dashboard has been generated:")
print(f"1. CSV summary saved to '{output_csv}'")
print(f"2. HTML dashboard saved to '{output_html}'")
print("Open the HTML file in your web browser to view the dashboard.")