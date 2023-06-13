# python-api-challenge

--------------------------------------------------------------------------------------------------------------
# WeatherPy

---

## Starter Code to Generate Random Geographic Coordinates and a List of Cities
# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import linregress

# Impor the OpenWeatherMap API key
from api_keys import weather_api_key

# Import citipy to determine the cities based on latitude and longitude
from citypy import citypy
### Generate the Cities List by Using the `citipy` Library
# Empty list for holding the latitude and longitude combinations
lat_lngs = []

# Empty list for holding the cities names
cities = []

# Range of latitudes and longitudes
lat_range = (-90, 90)
lng_range = (-180, 180)

# Create a set of random lat and lng combinations
lats = np.random.uniform(lat_range[0], lat_range[1], size=1500)
lngs = np.random.uniform(lng_range[0], lng_range[1], size=1500)
lat_lngs = zip(lats, lngs)

# Identify nearest city for each lat, lng combination
for lat_lng in lat_lngs:
    city = citipy.nearest_city(lat_lng[0], lat_lng[1]).city_name
    
    # If the city is unique, then add it to a our cities list
    if city not in cities:
        cities.append(city)

# Print the city count to confirm sufficient count
print(f"Number of cities in the list: {len(cities)}")
---
## Requirement 1: Create Plots to Showcase the Relationship Between Weather Variables and Latitude

### Use the OpenWeatherMap API to retrieve weather data from the cities list generated in the started code
# Set the API base URL
url = weather_api_key

# Define an empty list to fetch the weather data for each city
city_data = []

# Print to logger
print("Beginning Data Retrieval     ")
print("-----------------------------")

# Create counters
record_count = 1
set_count = 1

# Loop through all the cities in our list to fetch weather data
for i, city in enumerate(cities):
        
    # Group cities in sets of 50 for logging purposes
    if (i % 50 == 0 and i >= 50):
        set_count += 1
        record_count = 0

    # Create endpoint URL with each city
    city_url = url +"&q=" + urllib.request.pathname2url(city)
    
    # Log the url, record, and set numbers
    print("Processing Record %s of Set %s | %s" % (record_count, set_count, city))
    print(city_url)
    # Add 1 to the record count
    record_count += 1

    # Run an API request for each of the cities
    try:
        # Parse the JSON and retrieve data
        city_weather = requests.get(city_url).json()

        # Parse out latitude, longitude, max temp, humidity, cloudiness, wind speed, country, and date
        city_lat = city_weather["coord"]["lat"]
        city_lng = city_weather["coord"]["lng"]
        city_max_temp = city_weather["main"]["temp_max"]
        city_humidity = city_weather["main"]["humidity"]
        city_clouds = city_weather["clouds"]["all"]
        city_wind = city_weather["wind"]["speed"]
        city_country = city_weather["sys"]["country"]
        city_date = city_weather["dt"]

        # Append the City information into city_data list
        city_data.append({"City": city, 
                          "Lat": city_lat, 
                          "Lng": city_lng, 
                          "Max Temp": city_max_temp,
                          "Humidity": city_humidity,
                          "Cloudiness": city_clouds,
                          "Wind Speed": city_wind,
                          "Country": city_country,
                          "Date": city_date})

    # If an error is experienced, skip the city
    except:
        print("City not found. Skipping...")
        pass
              
# Indicate that Data Loading is complete 
print("-----------------------------")
print("Data Retrieval Complete      ")
print("-----------------------------")
# Convert the cities weather data into a Pandas DataFrame
city_data_df = pd.DataFrame(city_data_df)
#print(city_data_df)
#city_data_pd.head()

# Show Record Count
city_data_df.count()
# Display sample data
city_data_df.head()
# Export the City_Data into a csv
city_data_df.to_csv("output_data/cities.csv", index_label="City_ID")
# Read saved data
city_data_df = pd.read_csv("output_data/cities.csv", index_col="City_ID")

# Display sample data
city_data_df.head()
### Create the Scatter Plots Requested

#### Latitude Vs. Temperature
# Build scatter plot for latitude vs. temperature
plt.scatter(lats, max_temps, edgecolor = "black", linewidths = 1, marker = "o", alpha = 0.8, label = "Cities")

# Incorporate the other graph properties
plt.title(f"City Max Latitude vs. Temperature ({time.strftime(%x)})")
plt.xlabel("Latitude")
plt.ylabel("Max Temperature (C)")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig1.png")

# Show plot
plt.show()
#### Latitude Vs. Humidity
# Build the scatter plots for latitude vs. humidity
plt.scatter(lats, humidity, edgecolor = "black", linewidths = 1, marker = "o", alpha = 0.8, label = "Cities")

# Incorporate the other graph properties
plt.title(f"City Latitude vs. Humidity ({time.strftime(%x)})")
plt.xlabel("Latitude")
plt.ylabel("Humidity (%)")
plt.grid(True)


# Save the figure
plt.savefig("output_data/Fig2.png")

# Show plot
plt.show()
#### Latitude Vs. Cloudiness
# Build the scatter plots for latitude vs. cloudiness
plt.scatter(lats, cloudiness, edgecolor = "black", linewidths = 1, marker = "o", alpha = 0.8, label = "Cities")

# Incorporate the other graph properties
plt.title(f"City Latitude vs. Cloudiness ({time.strftime(%x)})")
plt.xlabel("Latitude")
plt.ylabel("Cloudiness (%)")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig3.png")

# Show plot
plt.show()
#### Latitude vs. Wind Speed Plot
# Build the scatter plots for latitude vs. wind speed
plt.scatter(lats, wind_speed, edgecolor = "black", linewidths = 1, marker = "o", alpha = 0.8, label = "Cities")

# Incorporate the other graph properties
plt.title(f"City Latitude vs. Wind Speed ({time.strftime(%x)})")
plt.xlabel("Latitude")
plt.ylabel("wind Speed (mph)")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig4.png")

# Show plot
plt.show()
---

## Requirement 2: Compute Linear Regression for Each Relationship

# Define a function to create Linear Regression plots
# regress_values = x_values * slope + intercept

# Create a DataFrame with the Northern Hemisphere data (Latitude >= 0)
northern_hemi_df = city_data_df.loc[pd.to_numeric(city_data_df["Lat"]) > 0, :]

# Display sample data
northern_hemi_df.head()
# Create a DataFrame with the Southern Hemisphere data (Latitude < 0)
southern_hemi_df = city_data_df.loc[pd.to_numeric(city_data_df["Lat"]) < 0, :]

# Display sample data
southern_hemi_df.head()
###  Temperature vs. Latitude Linear Regression Plot
# Linear regression on Northern Hemisphere
x_values = northern_hemi_df["Lat"]
y_values = northern_hemi_df["Max Temp"]
slope = linregress(x_values,y_values)
(slope, intercept, rvalue, pvalue, stderr)=linregress(x_values,y_values)
regress_values = x_values * slope + intercept
line_eq = "y =" + str(round(slope, 2))+ "x + " + str(round(intercept, 2))
plt.scatter(x_values,y_values)
plt.plot(x_values, regress_values, "-r")
plt.annotate(line_eq, (0,-20), fontsize = 15, color = "red")
plt.xlabel("Latitude")
plt.ylabel("Max Temp")
print(f'The r-value is: {rvalue}')

# Linear regression on Southern Hemisphere
x_values = southern_hemi_df["Lat"]
y_values = southern_hemi_df["Max Temp"]
slope = linregress(x_values,y_values)
(slope, intercept, rvalue, pvalue, stderr)=linregress(x_values,y_values)
regress_values = x_values * slope + intercept
line_eq = "y =" + str(round(slope, 2))+ "x + " + str(round(intercept, 2))
plt.scatter(x_values,y_values)
plt.plot(x_values, regress_values, "-r")
plt.annotate(line_eq, (0,-20), fontsize = 15, color = "red")
plt.xlabel("Latitude")
plt.ylabel("Max Temp")
print(f'The r-value is: {rvalue}')
print("Discussion about the linear relationship:")
print("The rvalue for the Northern Hemisphere is higher (closer to 1) than for the southern,")
print("suggesting that the the max temp is more relate to the latitude of the city in the Northern")
print("Hempisphere than in the Southern Hemisphere.")
**Discussion about the linear relationship:** YOUR RESPONSE HERE
### Humidity vs. Latitude Linear Regression Plot
# Northern Hemisphere
x_values = northern_hemi_df["Lat"]
y_values = northern_hemi_df["Humidity"]
slope = linregress(x_values,y_values)
(slope, intercept, rvalue, pvalue, stderr)=linregress(x_values,y_values)
regress_values = x_values * slope + intercept
line_eq = "y =" + str(round(slope, 2))+ "x + " + str(round(intercept, 2))
plt.scatter(x_values,y_values)
plt.plot(x_values, regress_values, "-r")
plt.annotate(line_eq, (0,-20), fontsize = 15, color = "red")
plt.xlabel("Latitude")
plt.ylabel("Humidity")
print(f'The r-value is: {rvalue}')
# Southern Hemisphere
x_values = southern_hemi_df["Lat"]
y_values = southern_hemi_df["Humidity"]
slope = linregress(x_values,y_values)
(slope, intercept, rvalue, pvalue, stderr)=linregress(x_values,y_values)
regress_values = x_values * slope + intercept
line_eq = "y =" + str(round(slope, 2))+ "x + " + str(round(intercept, 2))
plt.scatter(x_values,y_values)
plt.plot(x_values, regress_values, "-r")
plt.annotate(line_eq, (0,-20), fontsize = 15, color = "red")
plt.xlabel("Latitude")
plt.ylabel("Humidity")
print(f'The r-value is: {rvalue}')
print("Discussion about the linear relationship:")
print("The rvalue for both hemispheres are really small and suggest that they are not")
print("the humidity is not dependent on latitude for either. The Southern Hemisphere")
print("is still less so than the Northern Hemisphere.")
**Discussion about the linear relationship:** YOUR RESPONSE HERE
### Cloudiness vs. Latitude Linear Regression Plot
# Northern Hemisphere
x_values = northern_hemi_df["Lat"]
y_values = northern_hemi_df["Cloudiness"]
slope = linregress(x_values,y_values)
(slope, intercept, rvalue, pvalue, stderr)=linregress(x_values,y_values)
regress_values = x_values * slope + intercept
line_eq = "y =" + str(round(slope, 2))+ "x + " + str(round(intercept, 2))
plt.scatter(x_values,y_values)
plt.plot(x_values, regress_values, "-r")
plt.annotate(line_eq, (0,-20), fontsize = 15, color = "red")
plt.xlabel("Latitude")
plt.ylabel("Cloudiness")
print(f'The r-value is: {rvalue}')
# Southern Hemisphere
x_values = southern_hemi_df["Lat"]
y_values = southern_hemi_df["Cloudiness"]
slope = linregress(x_values,y_values)
(slope, intercept, rvalue, pvalue, stderr)=linregress(x_values,y_values)
regress_values = x_values * slope + intercept
line_eq = "y =" + str(round(slope, 2))+ "x + " + str(round(intercept, 2))
plt.scatter(x_values,y_values)
plt.plot(x_values, regress_values, "-r")
plt.annotate(line_eq, (0,-20), fontsize = 15, color = "red")
plt.xlabel("Latitude")
plt.ylabel("Cloudiness")
print(f'The r-value is: {rvalue}')
print("Discussion about the linear relationship:")
print("The rvalue for both hemispheres are similar again and suggest the cloudiness is not")
print("determined by latitude.")
**Discussion about the linear relationship:** YOUR RESPONSE HERE
### Wind Speed vs. Latitude Linear Regression Plot
# Northern Hemisphere
x_values = northern_hemi_df["Lat"]
y_values = northern_hemi_df["Wind Speed"]
slope = linregress(x_values,y_values)
(slope, intercept, rvalue, pvalue, stderr)=linregress(x_values,y_values)
regress_values = x_values * slope + intercept
line_eq = "y =" + str(round(slope, 2))+ "x + " + str(round(intercept, 2))
plt.scatter(x_values,y_values)
plt.plot(x_values, regress_values, "-r")
plt.annotate(line_eq, (0,-20), fontsize = 15, color = "red")
plt.xlabel("Latitude")
plt.ylabel("Wind Speed")
print(f'The r-value is: {rvalue}')
# Southern Hemisphere
x_values = southern_hemi_df["Lat"]
y_values = southern_hemi_df["Wind Speed"]
slope = linregress(x_values,y_values)
(slope, intercept, rvalue, pvalue, stderr)=linregress(x_values,y_values)
regress_values = x_values * slope + intercept
line_eq = "y =" + str(round(slope, 2))+ "x + " + str(round(intercept, 2))
plt.scatter(x_values,y_values)
plt.plot(x_values, regress_values, "-r")
plt.annotate(line_eq, (0,-20), fontsize = 15, color = "red")
plt.xlabel("Latitude")
plt.ylabel("Wind Speed")
print(f'The r-value is: {rvalue}')
print("Discussion about the linear relationship:")
print("The rvalue for the Southern Hemisphere is about twice the Northern Hemisphere's.")
print("This suggests the latitude is twice as likely to determine the wind speed in the")
print("Southern Hemisphere than in the Northern.")
**Discussion about the linear relationship:** YOUR RESPONSE HERE


--------------------------------------------------------------------------------------------------------------

# VacationPy
---

## Starter Code to Import Libraries and Load the Weather and Coordinates Data
# Dependencies and Setup
import hvplot.pandas
import pandas as pd
import requests

# Import API key
from api_keys import geoapify_key
# Load the CSV file created in Part 1 into a Pandas DataFrame
city_data_df = pd.read_csv("output_data/cities.csv")

# Display sample data
city_data_df.head()
---

### Step 1: Create a map that displays a point for every city in the `city_data_df` DataFrame. The size of the point should be the humidity in each city.
%%capture --no-display

map_plot = city_data_df.hvplot.points("Lng", "Lat", geo = True, tiles = "EsriImagery", size = "Humidity", frame_width = 800, frame_height = 600, scale = 0.5, color = "City")
# Display the map
map_plot
### Step 2: Narrow down the `city_data_df` DataFrame to find your ideal weather condition
# Narrow down cities that fit criteria and drop any results with null values
min_temp = city_data_df["Max Temp"] > 21
max_temp = city_data_df["Max Temp"] < 27
wind_speed = city_data_df["Wind Speed"] > 5
cloudiness_ = city_data_df["Cloudiness"] = 0

# Drop any rows with null values
great_weather_df = city_data_df[min_temp & max_temp & wind_speed & cloudiness]
# Display sample data
cleaned_weather_df = great_weather_df.dropna()
cleaned_weather_df
### Step 3: Create a new DataFrame called `hotel_df`.
# Use the Pandas copy function to create DataFrame called hotel_df to store the city, country, coordinates, and humidity
hotel_df = cleaned_weather_df.copy()

# Add an empty column, "Hotel Name," to the DataFrame so you can store the hotel found using the Geoapify API
hotel_df["Hotel Name"] = " "

# Display sample data
hotel_df
### Step 4: For each city, use the Geoapify API to find the first hotel located within 10,000 metres of your coordinates.
# Set parameters to search for a hotel
radius = 10000
params = {
    "categories": categories;
    "apiKey": geoapify_key
}

# Print a message to follow up the hotel search
print("Starting hotel search")

# Iterate through the hotel_df DataFrame
for index, row in hotel_df.iterrows():
    # get latitude, longitude from the DataFrame
    lng = hotel.df.loc[index, "Lng"]
    lat = hotel.df.loc[index, "Lat"]
    
    # Add filter and bias parameters with the current city's latitude and longitude to the params dictionary
    params["filter"] = f"circle: {lng}, {lat}, {radius}"
    params["bias"] = f"proximity:{lng}, {lat}"
    
    # Set base URL
    base_url = "https://api.geoapify.com/v2/places"


    # Make and API request using the params dictionaty
    name_address = requests.get(base_url, params= params)
    #print(name_address.url)
    
    # Convert the API response to JSON format
    name_address = name_address.json()
    
    # Grab the first hotel from the results and store the name in the hotel_df DataFrame
    try:
        hotel_df.loc[index, "Hotel Name"] = name_address["features"][0]["properties"]["name"]
    except (KeyError, IndexError):
        # If no hotel is found, set the hotel name as "No hotel found".
        hotel_df.loc[index, "Hotel Name"] = "No hotel found"
        
    # Log the search results
    print(f"{hotel_df.loc[index, 'City']} - nearest hotel: {hotel_df.loc[index, 'Hotel Name']}")

# Display sample data
hotel_df
### Step 5: Add the hotel name and the country as additional information in the hover message for each city in the map.
%%capture --no-display

# Configure the map plot
hotels_map = hotel_df.hvplot.points("Lng", "Lat", geo = True, tiles = "EsriImagery", frame_width 800, frame_height = 600, scale = 0.5, color = "City")

# Display the map
hotels_map
