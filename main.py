from Sub_Functions.Analysis import Analysis
from Sub_Functions.Plot import Plot_Results
from Sub_Functions.Read_data import read_and_preprocess_data
import os
import numpy as np
from Sub_Functions.myplot import ALL_GRAPH_PLOT

Data=["Zea_mays","Solanum_pennellii"]  # List of data to be analyzed

VVV=input("DO YOU WANT TO CONTINUE WITH FULL EXECUTION(type Yes/No (Case Sensitive)):") # THis is the point which asks the user if they want to continue with full execution or not.

if VVV=="Yes":  # if excecution is true then the code will execute

        for i in range(len(Data)):  # This is the loop over the
                # read_and_preprocess_data(data=Data[i])  #This line reads,preprocess and stores the data

                TP=Analysis(Data[i])    # This line creates an object of Analysis class and also passes the data as an argument

                TP.TP_Analysis()       # This line calls TP_Analysis and performs

                TP.Performance_Analysis()  # This line calls Performance_Analysis and performs

                # TP.KF_Analysis()     #This line performs the KF Analysis

                PL=ALL_GRAPH_PLOT()  # This line creates an object of Plot_Results class

                PL.GRAPGH_RESULT(DB=Data[i])  # This line calls AnalysisResult and plots the results


else:    #  if execution is false then the code will not execute

        Data = ["Solanum_pennellii"] # List of data to be analyzed

        for i in range(len(Data)): # This loop will iterate through the list of data to be analyzed

                PL = ALL_GRAPH_PLOT()  # This line creates an object of Plot_Results class

                PL.GRAPGH_RESULT(DB=Data[i])  # This line calls AnalysisResult and plots the results

                # PL=Plot_Results()

                # PL.AnalysisResult(DB=Data[i])








