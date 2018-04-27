#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: dorisxx
"""


#python script for collecting weather data through wolfram api, built on python2.7
#change appid to your appid
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import itertools

appid = 'XXX'

def get_data(y, La, Lo):
    year = int(y)
    Lat = La
    Lon = Lo
    url_1 = "http://api.wolframalpha.com/v2/query?appid="+appid
    url = url_1+"&input=weather+%s+Latitude%%3D%s+Longitude%%3D+%s%%3f" %(year,Lat,Lon)
    r = requests.get(url)
    soup = BeautifulSoup(r.text,"lxml")
    
    
    #get pressure information hPa
    try:
        pressure = soup.find("subpod",attrs={"title":"Pressure"}).find("plaintext").string
        pressure_num = float(re.findall('\d+\.*\d*', pressure)[0])
    except:
        pressure_num = None
    
    #get maximum precipitation inch
    try:
        precipitation = soup.find("subpod",attrs={"title":"Precipitation amount"}).find("plaintext").string
        precipitation_num = float(re.findall('\d+\.*\d*', precipitation)[0])
    except:
        precipitation_num = None
        
    #get temperature F
    try:
        temperature = soup.find("subpod",attrs={"title":"Temperature"}).find("plaintext").string
        temperature_low = float(re.findall('-\d+\.*\d*|\d+\.*\d*', temperature)[0])
        temperature_average_high = float(re.findall('-\d+\.*\d*|\d+\.*\d*', temperature)[3])
        temperature_average_low = float(re.findall('-\d+\.*\d*|\d+\.*\d*', temperature)[4])
        temperature_high = float(re.findall('-\d+\.*\d*|\d+\.*\d*', temperature)[5])
    except:
        temperature_low = None
        temperature_average_high = None
        temperature_average_low = None
        temperature_high = None
        
    
    #get humidity  %
    try:
        humidity= soup.find("subpod",attrs={"title":"Humidity"}).find("plaintext").string
        humidity_num= float(re.findall('-\d+\.*\d*|\d+\.*\d*', humidity)[0])
    except:
        humidity_num= None
        
    
    #get wind speed mph
    try:
        wind= soup.find("subpod",attrs={"title":"Wind speed"}).find("plaintext").string
        wind_average = float(re.findall('-\d+\.*\d*|\d+\.*\d*', wind)[0])
        wind_high = float(re.findall('-\d+\.*\d*|\d+\.*\d*', wind)[1])
    except:
        wind_average = None
        wind_high = None
        
    temp = [temperature_low,temperature_average_high,temperature_average_low,temperature_high]
    wind_data = [wind_average,wind_high]
    data = [pressure_num,precipitation_num,temp,humidity_num,wind_data]
    
    return data

def main():
    data_test = pd.read_csv("county_texas.csv")
    col_1 = ['county','Lat','Lon']
    col_2 = ['pressure/hPa','max_precipitation','humidity']
    col_3 = ['temperature_low','temperature_high','temperature_average_high','temperature_average_low',]
    col_4 = ['wind_average','wind_high']
    col = col_1+col_2+col_3+col_4
    output = pd.DataFrame(columns = col)
    for index, row in data_test.iterrows():
        weather = get_data(2017,row[1],row[2])
        l1 = [weather[0],weather[1],weather[3]]
        l2 = [weather[2][0],weather[2][3],weather[2][1],weather[2][2]]
        l3 = [weather[4][0],weather[4][1]]
    
        output.loc[index] = list(itertools.chain(row,l1, l2, l3))
    output.to_csv("data_collected.csv",sep = ',', header = True, index = False)
    
if __name__ == "__main__":
    main()

