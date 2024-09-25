import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from warnings import filterwarnings
import webbrowser

# Ignore warnings
filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('C:/Users/Lenovo/zomato/Cleaned_data.csv')

print("Columns in DataFrame:", data.columns)

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Filter data for restaurants with a rating of 4 or more and cost less than 500
df_new = data[(data['rate'] >= 4) & (data['cost'] <= 500)]
print(f"Unique restaurant names: {len(df_new['name'].unique())}")

# Group by location to count unique restaurant names
location_counts = df_new.groupby('location')['name'].nunique().reset_index()
location_counts.columns = ['Name', 'Count']

# Prepare location DataFrame for geocoding
locations = pd.DataFrame(location_counts['Name'].unique(), columns=['Name'])
locations['New_Name'] = 'Bangalore ' + locations['Name']

# Geocode locations using Nominatim
lat_lon = []
geolocator = Nominatim(user_agent="app")
for loc in locations['Name']:
    location = geolocator.geocode(loc)
    if location is None:
        lat_lon.append((np.nan, np.nan))
    else:
        lat_lon.append((location.latitude, location.longitude))

# Store latitude and longitude in the DataFrame
locations['Geo_Loc'] = lat_lon
locations[['Latitude', 'Longitude']] = pd.DataFrame(locations['Geo_Loc'].tolist(), index=locations.index)

# Merge the location data with the restaurant counts
heatmap_data = location_counts.merge(locations, left_on='Name', right_on='Name').dropna()

# Generate a base map centered around Bangalore
def generate_base_map(default_location=[12.97, 77.59], default_zoom_start=12):
    return folium.Map(location=default_location, zoom_start=default_zoom_start)

# Create and add heatmap to the base map
basemap = generate_base_map()
HeatMap(heatmap_data[['Latitude', 'Longitude', 'Count']].values.tolist(), radius=15).add_to(basemap)

# Save the map to an HTML file
basemap.save('restaurant_heatmap.html')

# Optionally, open the saved map in the web browser
webbrowser.open('restaurant_heatmap.html')