import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# read all the csv files from the folder "HYP_T_Data_Files/OneDrive_2023-01-15 (1)/All participants/corepill"

# all the csv files in the folder end with .csv

location = "../HYP_T_Data_Files/OneDrive_2023-01-15 (1)/All participants/corepill"

csv_files = [csv_file for csv_file in os.listdir(location) if csv_file.endswith(".csv")]


# read all the csv files and remove the first 5 rows from each csv file and append them to a list

df_list = []

for csv_file in csv_files:
    df = pd.read_csv(location + "/" + csv_file, skiprows=5)
    df_list.append(df)

# check missing values in each dataframe 4th column vs total number of rows in each dataframe

for df in df_list:
    print(df.iloc[:, 4].isna().sum() , len(df))


# set 3rd as time "12:31:16 PM" and index for each dataframe

for df in df_list:
    df.iloc[:, 3] = pd.to_datetime(df.iloc[:, 3])
    df.set_index(df.iloc[:, 3], inplace=True)




# replace missing values with linear interpolation and check again

for df in df_list:
    df.iloc[:, 4] = df.iloc[:, 4].interpolate(method='linear', limit_direction='forward')
    print(df.iloc[:, 4].isna().sum() , len(df))


# plot the data with time (index) as x axis and temperature as y axis

for df in df_list:
    plt.plot(df.index, df.iloc[:, 4])
    plt.show()


# create a new folder called "corepill_preprocessed" and save all the dataframes to csv files

os.mkdir("../HYP_T_Data_Files/OneDrive_2023-01-15 (1)/All participants/corepill_preprocessed")

for i in range(0, len(df_list)):
    df_list[i].to_csv("../HYP_T_Data_Files/OneDrive_2023-01-15 (1)/All participants/corepill_preprocessed/" + csv_files[i])


