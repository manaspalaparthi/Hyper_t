import numpy as np
import pandas as pd
import os
import datetime

# read a txt file from a folder and return a dataframe

def read_files(folder_name:str):
    """Reads all files in a folder and returns as dataframes"""
    files = os.listdir(folder_name)
    # select file ending with .txt
    files = [file for file in files if file.endswith('.txt')]
    #read a txt file using open
    with open(folder_name + "/" + files[0], 'r') as f:
        df = pd.read_csv(f, sep=" ", header=None, on_bad_lines='skip')
    return df , files[0]


data, file = read_files("../HYP_T_Data_Files/OneDrive_2023-01-15 (1)/All participants/HRV/HYP_T 01")

data.columns = ['HRV']

# extract date and time from the file name

def extract_date_time(file_name):
    """Extracts date and time from the file name rr-University_Deakin_2022-10-24_11-14-54.txt"""
    date_time = file_name.split("_")[2:4]
    date_time = " ".join(date_time)
    date_time = date_time.replace(".txt", "")
    date_time = datetime.datetime.strptime(date_time, '%Y-%m-%d %H-%M-%S')
    return date_time

print(extract_date_time(file))
# split the row into 2 columns by space

def split_row(dataframe):
    """Splits the row into 2 columns by space"""
    dataframe = dataframe['HRV'].str.split(" ", n=1, expand=True)

    return dataframe

data = split_row(data)

# rename the columns

data.columns = ['second', 'HRV']

# each row add datetime to the second column

def add_datetime(dataframe, time):
    """Adds datetime to the second column, seconds is in float"""

    dataframe['second'] = dataframe['second'].astype(float)
    dataframe['datetime'] = dataframe['second'].apply(lambda x: time + datetime.timedelta(seconds=x))
    # drop the second column
    dataframe = dataframe.drop(columns=['second'])
    return dataframe


data = add_datetime(data, extract_date_time(file))

# convert HRV to float

data['HRV'] = data['HRV'].astype(float)

print(data)

