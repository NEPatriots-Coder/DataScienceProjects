import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
import os 
import openpyxl

print("Starting the Biden Executive Order Location Mapper...")

# --- Step 1: Load and Validate Data from Excel File ---

# Define the path to your Excel file.
# The 'r' before the string makes it a "raw string", which helps handle backslashes in Windows paths.
file_path = r'C:\Users\lamarw\Desktop\ComputerScience\DataScienceProjects\Biden_Locations.xlsx'

# Check if the file actually exists before trying to read it
if not os.path.exists(file_path):
    print(f"--- ERROR ---")
    print(f"The file was not found at the specified path: {file_path}")
    print("Please make sure the path is correct and the file is in the right place.")
    exit() # Stop the script if the file isn't found

# Define the columns we absolutely need from the Excel file
required_columns = ['date', 'action', 'location1', 'location1_summary', 'in_washington_dc', 'location2','location2_summary']

try:
    print(f"\nReading data from Excel file: {file_path}")
    # Use pandas to read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    print(f"Successfully loaded {len(df)} rows from the file.")

    # Validate that all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"--- ERROR ---")
        print(f"The Excel file is missing the following required columns: {', '.join(missing_columns)}")
        print("Please check your column headers in the Excel file.")
        exit()

except Exception as e:
    print(f"--- ERROR ---")
    print(f"An error occurred while reading the Excel file: {e}")
    exit()


# --- Step 2: Geocode Locations to Get Coordinates ---

# Initialize the geocoder (Nominatim is a free service from OpenStreetMap)
geolocator = Nominatim(user_agent="biden_location_mapper_v2")

# To avoid overwhelming the free service, we add a 1-second delay between requests
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

print("\nGeocoding locations... (This may take a moment depending on the number of rows)")

# Create a function to get coordinates, handling potential errors
def get_coordinates(location_str):
    try:
        location = geocode(location_str)
        if location:
            print(f"  > Found coordinates for: {location_str}")
            return (location.latitude, location.longitude)
        else:
            print(f"  > Could not find coordinates for: {location_str}")
            return (None, None)
    except Exception as e:
        print(f"  > Error geocoding {location_str}: {e}")
        return (None, None)

# Apply the function to the 'location2' column to get coordinates
df[['latitude', 'longitude']] = df['location2'].apply(get_coordinates).apply(pd.Series)

# Report on geocoding success
successful_geocodes = df['latitude'].notna().sum()
print(f"\nGeocoding complete. Successfully found coordinates for {successful_geocodes} out of {len(df)} locations.")


# --- Step 3: Create the Interactive Map ---

print("\nCreating the interactive map...")
# Create a base map, centered on the USA
map_center = [39.8283, -98.5795]
biden_map = folium.Map(location=map_center, zoom_start=4, tiles="CartoDB positron")


# --- Step 4: Add Markers with Popups to the Map ---

# Iterate through our DataFrame and add a marker for each event
for index, row in df.iterrows():
    # Only add a marker if we have valid coordinates
    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
        
        # Customize the popup text with HTML for better formatting
        popup_html = f"""
        <b>Date:</b> {row['date']}<br>
        <b>Location:</b> {row['location2']}<br>
        <hr style="margin: 3px;">
        <b>Summary:</b><br>{row['location1_summary']}
        """
        
        # Choose marker color based on the 'in_washington_dc' flag
        # This works correctly if the column contains TRUE/FALSE values
        if row['in_washington_dc']:
            marker_color = 'red'
            marker_icon = 'home'
        else:
            marker_color = 'blue'
            marker_icon = 'briefcase'
            
        # Create the popup and marker
        popup = folium.Popup(popup_html, max_width=300)
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup,
            tooltip=f"<b>{row['location2']}</b><br>Click to see details",
            icon=folium.Icon(color=marker_color, icon=marker_icon, prefix='fa')
        ).add_to(biden_map)

# --- Step 5: Save the Map to an HTML File ---

output_filename = 'biden_locations_map_from_file.html'
biden_map.save(output_filename)

print(f"\n🎉 Success! Map has been saved to '{output_filename}'")
print("Open this file in your web browser to see your data visualized on an interactive map.")