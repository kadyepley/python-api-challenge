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

# Import the OpenWeatherMap API key
from api_keys import weather_api_key

# Import citipy to determine the cities based on latitude and longitude
from citipy import citipy

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
url = "https://api.openweathermap.org/data/2.5/weather?units=metric&APPID=" + weather_api_key
print(url)
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
    city_url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric'
    
    # Log the url, record, and set numbers
    print("Processing Record %s of Set %s | %s" % (record_count, set_count, city))
    #print(city_url)
    
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
    except KeyError:
        print("City not found. Skipping...")
        pass
              
# Indicate that Data Loading is complete 
print("-----------------------------")
print("Data Retrieval Complete      ")
print("-----------------------------")
# Convert the cities weather data into a Pandas DataFrame
city_data_df = pd.DataFrame(city_data)
#print(city_data_df)
#city_data_pd.head()

# Show Record Count
city_data_df.count()
# Display sample data
city_data_df.head()
# Export the City_Data into a csv
city_data_df.to_csv("output_data_file/cities.csv", index_label="City_ID")
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


---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
Cell In[3], line 2
      1 # Load the CSV file created in Part 1 into a Pandas DataFrame
----> 2 city_data_df = pd.read_csv("output_data/cities.csv")
      4 # Display sample data
      5 city_data_df.head()

File c:\Users\kadye\anaconda3\envs\EdX\lib\site-packages\pandas\util\_decorators.py:211, in deprecate_kwarg.._deprecate_kwarg..wrapper(*args, **kwargs)
    209     else:
    210         kwargs[new_arg_name] = new_arg_value
--> 211 return func(*args, **kwargs)

File c:\Users\kadye\anaconda3\envs\EdX\lib\site-packages\pandas\util\_decorators.py:331, in deprecate_nonkeyword_arguments..decorate..wrapper(*args, **kwargs)
    325 if len(args) > num_allow_args:
    326     warnings.warn(
    327         msg.format(arguments=_format_argument_list(allow_args)),
    328         FutureWarning,
    329         stacklevel=find_stack_level(),
    330     )
--> 331 return func(*args, **kwargs)

File c:\Users\kadye\anaconda3\envs\EdX\lib\site-packages\pandas\io\parsers\readers.py:950, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
    935 kwds_defaults = _refine_defaults_read(
    936     dialect,
...
    863     else:
    864         # Binary mode
    865         handle = open(handle, ioargs.mode)

FileNotFoundError: [Errno 2] No such file or directory: 'output_data/cities.csv'
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[4], line 1
----> 1 map_plot = city_data_df.hvplot.points("Lng", "Lat", geo = True, tiles = "EsriImagery", size = "Humidity", frame_width = 800, frame_height = 600, scale = 0.5, color = "City")
      2 # Display the map
      3 map_plot

NameError: name 'city_data_df' is not defined
City_ID	City	Lat	Lng	Max Temp	Humidity	Cloudiness	Wind Speed	Country	Date
45	45	kapaa	22.0752	-159.3190	22.99	84	0	3.60	US	1666108257
51	51	hilo	19.7297	-155.0900	26.27	83	0	2.57	US	1666108260
63	63	banda	25.4833	80.3333	24.62	52	0	2.68	IN	1666108268
81	81	makakilo city	21.3469	-158.0858	21.66	81	0	2.57	US	1666108282
152	152	kahului	20.8947	-156.4700	23.80	60	0	3.09	US	1666108246
197	197	gat	31.6100	34.7642	24.38	100	0	3.69	IL	1666108356
211	211	laguna	38.4210	-121.4238	21.67	79	0	2.06	US	1666108364
240	240	tikaitnagar	26.9500	81.5833	23.56	59	0	0.35	IN	1666108378
265	265	san quintin	30.4833	-115.9500	21.20	74	0	1.37	MX	1666108394
340	340	santa rosalia	27.3167	-112.2833	24.62	56	0	0.74	MX	1666108436
363	363	narwar	25.6500	77.9000	22.35	55	0	1.29	IN	1666108449
375	375	port hedland	-20.3167	118.5667	21.03	73	0	3.09	AU	1666108455
381	381	roebourne	-20.7833	117.1333	23.48	65	0	2.95	AU	1666108458
391	391	saint-francois	46.4154	3.9054	23.69	57	0	4.12	FR	1666108465
409	409	capoterra	39.1763	8.9718	24.84	71	0	3.60	IT	1666108477
421	421	stolac	43.0844	17.9575	24.88	68	0	0.80	BA	1666108483
516	516	guerrero negro	27.9769	-114.0611	23.17	68	0	0.89	MX	1666108537
City	Country	Lat	Lng	Humidity	Hotel Name
45	kapaa	US	22.0752	-159.3190	84	
51	hilo	US	19.7297	-155.0900	83	
63	banda	IN	25.4833	80.3333	52	
81	makakilo city	US	21.3469	-158.0858	81	
152	kahului	US	20.8947	-156.4700	60	
197	gat	IL	31.6100	34.7642	100	
211	laguna	US	38.4210	-121.4238	79	
240	tikaitnagar	IN	26.9500	81.5833	59	
265	san quintin	MX	30.4833	-115.9500	74	
340	santa rosalia	MX	27.3167	-112.2833	56	
363	narwar	IN	25.6500	77.9000	55	
375	port hedland	AU	-20.3167	118.5667	73	
381	roebourne	AU	-20.7833	117.1333	65	
391	saint-francois	FR	46.4154	3.9054	57	
409	capoterra	IT	39.1763	8.9718	71	
421	stolac	BA	43.0844	17.9575	68	
516	guerrero negro	MX	27.9769	-114.0611	68	
Starting hotel search
kapaa - nearest hotel: Pono Kai Resort
hilo - nearest hotel: Dolphin Bay Hotel
banda - nearest hotel: #acnindiafy21
makakilo city - nearest hotel: Embassy Suites by Hilton Oahu Kapolei
kahului - nearest hotel: Maui Seaside Hotel
gat - nearest hotel: No hotel found
laguna - nearest hotel: Holiday Inn Express & Suites
tikaitnagar - nearest hotel: No hotel found
san quintin - nearest hotel: Jardines Hotel
santa rosalia - nearest hotel: Hotel del Real
narwar - nearest hotel: No hotel found
port hedland - nearest hotel: The Esplanade Hotel
roebourne - nearest hotel: No hotel found
saint-francois - nearest hotel: Chez Lily
capoterra - nearest hotel: Rosa Hotel
stolac - nearest hotel: Bregava
guerrero negro - nearest hotel: Plaza sal paraiso
City	Country	Lat	Lng	Humidity	Hotel Name
45	kapaa	US	22.0752	-159.3190	84	Pono Kai Resort
51	hilo	US	19.7297	-155.0900	83	Dolphin Bay Hotel
63	banda	IN	25.4833	80.3333	52	#acnindiafy21
81	makakilo city	US	21.3469	-158.0858	81	Embassy Suites by Hilton Oahu Kapolei
152	kahului	US	20.8947	-156.4700	60	Maui Seaside Hotel
197	gat	IL	31.6100	34.7642	100	No hotel found
211	laguna	US	38.4210	-121.4238	79	Holiday Inn Express & Suites
240	tikaitnagar	IN	26.9500	81.5833	59	No hotel found
265	san quintin	MX	30.4833	-115.9500	74	Jardines Hotel
340	santa rosalia	MX	27.3167	-112.2833	56	Hotel del Real
363	narwar	IN	25.6500	77.9000	55	No hotel found
375	port hedland	AU	-20.3167	118.5667	73	The Esplanade Hotel
381	roebourne	AU	-20.7833	117.1333	65	No hotel found
391	saint-francois	FR	46.4154	3.9054	57	Chez Lily
409	capoterra	IT	39.1763	8.9718	71	Rosa Hotel
421	stolac	BA	43.0844	17.9575	68	Bregava
516	guerrero negro	MX	27.9769	-114.0611	68	Plaza sal paraiso
