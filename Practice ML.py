#practice aqi

from prophet import Prophet
from prophet.diagnostics import *
import pandas as pd
import datetime as dt
import numpy as np
import warnings
warnings.simplefilter("ignore", category = FutureWarning)
#the seperators are ; and the decimal is ,
data = pd.read_csv("/AirQualityUCI.csv", sep = ";", decimal = ",")
#we have two na columns, getting rid of them by selecting a range that excludes them
data = data.iloc[:,:-2]
#we've garbage rows beyond 9357, trimming them out using head, the datasets mention that it only goes to 9357.
data = data.head(9357)
#converting date to the right format, the day is first or set the format to mixed
#datetime.strftime is used to set it to the desired format.
data["Date"] = pd.to_datetime(data["Date"], format = "mixed").dt.strftime("%Y-%m-%d")
print(data.info())
#In this dataset, the null value is set to -200
#we need to check to see the number of null values and set them to na
print(data.isin([-200]).sum())
#replacing the null values to na using numpy.nan variable
data.replace(to_replace = -200, value = np.nan)
#count the number of null values with isnull
print(data.isnull().sum())
#now replacing the null values with mean value, since inplace is set to true, then the replacing is done on the current string.
data.fillna(data.mean(numeric_only = True), inplace = True)

#formatting time to change the dot to : as the seperator.
data["Time"] = data["Time"].apply(lambda x:x.replace(".",":"))

#create ds to store the date and time

#encountered and error while concating, it seems it doesn't allows reindexing on an axis with duplicate labels
#try creating a new dataframe to store this

#creating a new dataframe to be used for the Prophet model.
data1 = pd.DataFrame()

#Adjoining the Date and Time values and storing them in ds in dataframe data1
data1["ds"] = data["Date"].astype(str) + " " + data["Time"].astype(str)
#Setting y to Relative humidity
data1["y"] = data["RH"]

#Creating model
model = Prophet()
#fitting data in model
model.fit(data1)
#Creating future dataframe
future = model.make_future_dataframe(periods = 365, freq = "h")
#predicting forecast based on future dataframe
forecast = model.predict(future)
#plotting the model
fig = model.plot_components(forecast)
