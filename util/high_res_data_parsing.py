import numpy as np
import pandas as pd
import os


def read_files(folder_name:str):
    """Reads all files in a folder and returns a list of dataframes"""
    files = os.listdir(folder_name)
    #read a csv file
    df = pd.read_csv('data.csv')

def filter_tempdata(dataframe):
    temp_data = dataframe[12:]

    # change the column names
    temp_data.columns = ['frame_number', 'temp_C', "temp_F"]

    # reset index
    temp_data = temp_data.reset_index(drop=True)

    # remove last column
    temp_data = temp_data.drop(columns=['temp_F'])

    # remove "Frame" from the frame number and convert to int
    temp_data['frame_number'] = temp_data['frame_number'].str.replace('Frame', '').astype(int)

    #convert temp_C to float
    temp_data['temp_C'] = temp_data['temp_C'].astype(float)

    return temp_data


def main():
    df = pd.read_csv('../HYP_T_Data_Files/OneDrive_2023-01-15 (1)/All participants/High Res Thermal Camera/HYT03/rec_0001.csv')

    df = filter_tempdata(df)

    print(df)

    # select files ending with .csv
    #files = [file for file in files if file.endswith('.csv')]







if __name__ == "__main__":
    main()

