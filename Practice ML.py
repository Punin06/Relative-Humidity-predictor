#practice aqi

from prophet import Prophet
from prophet.diagnostics import *
import pandas as pd
import datetime as dt
import numpy as np
import warnings
warnings.simplefilter("ignore", category = FutureWarning)
#the seperators are ; and the decimal is ,
data = pd.read_csv("/Users/nipunp/Desktop/air+quality/AirQualityUCI.csv", sep = ";", decimal = ",")
#we have two na columns, getting rid of them by selecting a range that excludes them
data = data.iloc[:,:-2]
#we've garbage rows beyond 9357, trim them out using head, the datasets mention that it only goes to 9357 rest is missing values
data = data.head(9357)
#without this I was finding null values for time and date, and I wasn't able to change them until I removed those garbage rows
#converting date to the right format, the day is first or set the format to mixed
#datetime.strftime is used to set it to the desired format. Y is capital.
data["Date"] = pd.to_datetime(data["Date"], format = "mixed").dt.strftime("%Y-%m-%d")
#In time we got . instead : need to replace that, we use apply() as it works on all values
#data["Time"] = data["Time"].apply(lambda x:x.replace(",",":"))
#there's a type error in time, check the info of dataframe to see the types of the labels
print(data.info())
#time is an object, perhaps the issues lies in the null values
#remember in this dataset, the null value is set to -200
#we need to check to see the number of null values and set them to na
print(data.isin([-200]).sum())
#replacing the null values to na using numpy.nan variable
data.replace(to_replace = -200, value = np.nan)
#count the number of null values with isnull
print(data.isnull().sum())
#now replace the null values with mean value, mean doesn't have inplace, use fillna, it's made to replace null, if inplace is set to true, then the replacing is done on the current string 
data.fillna(data.mean(numeric_only = True), inplace = True)

data["Time"] = data["Time"].apply(lambda x:x.replace(".",":"))

#create ds to store the date and time

#encountered and error while concating, it seems it doesn't allows reindexing on an axis with duplicate labels
#try creating a new dataframe to store this
#creating a new dataframe solves the problem but concating and setting the axis to 1 doesn't let's me to join multiple columns to one column

data1 = pd.DataFrame()

#data1["ds"] = pd.concat([data["Date"],data["Time"]]) you can exclude this one
#all the values are in one single column, but to make them adjacent to each other
data1["ds"] = data["Date"].astype(str) + " " + data["Time"].astype(str)

data1["y"] = data["RH"]


print(data.isnull().sum())
print(data.head())
print(data1.tail())

model = Prophet()
model.fit(data1)
future = model.make_future_dataframe(periods = 365, freq = "h")
forecast = model.predict(future)
fig = model.plot_components(forecast)

e = cross_validation(model, initial = "365 hours", period = "28 hours", horizon = "120 hours" )
df_p = performance_metrics(e)
print (df_p.head())


