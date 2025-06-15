import numpy as np
import pandas as pd

def Concat_epochs(DB):
    A = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_40percent_epoch100.npy")
    B = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_50percent_epoch100.npy")
    C = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_60percent_epoch100.npy")
    D = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_70percent_epoch100.npy")
    E = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_80percent_epoch100.npy")
    F = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_90percent_epoch100.npy")


    AA = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_40percent_epoch200.npy")
    BB = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_50percent_epoch200.npy")
    CC = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_60percent_epoch200.npy")
    DD = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_70percent_epoch200.npy")
    EE = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_80percent_epoch200.npy")
    FF = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_90percent_epoch200.npy")


    AAA = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_40percent_epoch300.npy")
    BBB = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_50percent_epoch300.npy")
    CCC = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_60percent_epoch300.npy")
    DDD = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_70percent_epoch300.npy")
    EEE = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_80percent_epoch300.npy")
    FFF = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_90percent_epoch300.npy")

    AAAA = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_40percent_epoch400.npy")
    BBBB = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_50percent_epoch400.npy")
    CCCC = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_60percent_epoch400.npy")
    DDDD = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_70percent_epoch400.npy")
    EEEE = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_80percent_epoch400.npy")
    FFFF = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_90percent_epoch400.npy")

    AAAAA = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_40percent_epoch500.npy")
    BBBBB = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_50percent_epoch500.npy")
    CCCCC = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_60percent_epoch500.npy")
    DDDDD = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_70percent_epoch500.npy")
    EEEEE = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_80percent_epoch500.npy")
    FFFFF = np.load(f"Analysis/Performance_Analysis/Zea_mays/metrics_90percent_epoch500.npy")


    A1=np.stack([A,B,C,D,E,F],axis=1)
    A2=np.stack([AA,BB,CC,DD,EE,FF],axis=1)
    A3=np.stack([AAA,BBB,CCC,DDD,EEE,FFF],axis=1)
    A4=np.stack([AAAA,BBBB,CCCC,DDDD,EEEE,FFFF],axis=1)
    A5=np.stack([AAAAA,BBBBB,CCCCC,DDDDD,EEEEE,FFFFF],axis=1)

    np.save(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_100.npy",A1)
    np.save(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_200.npy", A2)
    np.save(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_300.npy", A3)
    np.save(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_400.npy", A4)
    np.save(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_500.npy", A5)

    A11=np.load(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_100.npy")
    A22=np.load(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_200.npy")
    A33=np.load(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_300.npy")
    A44=np.load(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_400.npy")
    A55=np.load(f"Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_500.npy")


    return A11,A22,A33,A44,A55

# def metrics_concat(DB):
# A11 = np.load(f"../Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_100.npy")
# A22 = np.load(f"../Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_200.npy")
# A33 = np.load(f"../Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_300.npy")
# A44 = np.load(f"../Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_400.npy")
# A55 = np.load(f"../Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_500.npy")
#
# print(A11)
# A1=np.stack([A11[0],A22[0],A33[0],A44[0],A55[0]],axis=0)
# A2=np.stack([A11[1],A22[1],A33[1],A44[1],A55[1]],axis=0)
# A3=np.stack([A11[2],A22[2],A33[2],A44[2],A55[2]],axis=0)
# A4=np.stack([A11[3],A22[3],A33[3],A44[3],A55[3]],axis=0)
# A5=np.stack([A11[4],A22[4],A33[4],A44[4],A55[4]],axis=0)
# A6=np.stack([A11[5],A22[5],A33[5],A44[5],A55[5]],axis=0)
#
#
# print(A1)



# A = pd.read_csv(f"../Analysis/KF_Analysis/Zea_mays/ACC_2.csv").values
# B = pd.read_csv(f"../Analysis/KF_Analysis/Zea_mays/SEN_2.csv").values
# C = pd.read_csv(f"../Analysis/KF_Analysis/Zea_mays/SPE_2.csv").values
# D = pd.read_csv(f"../Analysis/KF_Analysis/Zea_mays/F1score_2.csv").values
# E = pd.read_csv(f"../Analysis/KF_Analysis/Zea_mays/REC_2.csv").values
# F = pd.read_csv(f"../Analysis/KF_Analysis/Zea_mays/PRE_2.csv").values
#
# print(A.shape)
# print(B.shape)
# print(C.shape)
# print(D.shape)
# print(E.shape)
# print(F.shape)

import os
import numpy as np
import pandas as pd
import os


def Convert_Comp_To_csv(DB):
    base_path = f"../Analysis/Comparative_Analysis/{DB}"

    file_names = [
        "ACC_1.npy",
        "SEN_1.npy",
        "SPE_1.npy",
        "PRE_1.npy",
        "REC_1.npy",
        "F1score_1.npy"
    ]

    for file_name in file_names:
        npy_path = os.path.join(base_path, file_name)
        csv_path = os.path.join(base_path, file_name.replace(".npy", ".csv"))

        # Load and convert
        data = np.load(npy_path)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        print(f"Saved: {csv_path}")


# Example usage
# Convert_Comp_To_csv("Zea_mays")

import numpy as np
import pandas as pd
import os

def Convert_Comp_To_csv(DB):
    base_path = f"../Analysis/Comparative_Analysis/{DB}"

    metric_data = {
        "ACC_1": [
            [0.65, 0.68, 0.71, 0.73, 0.75, 0.77],
            [0.66, 0.69, 0.72, 0.74, 0.76, 0.78],
            [0.64, 0.67, 0.70, 0.73, 0.75, 0.76],
            [0.67, 0.69, 0.71, 0.74, 0.76, 0.77],
            [0.65, 0.68, 0.70, 0.72, 0.74, 0.76],
            [0.70, 0.73, 0.76, 0.78, 0.81, 0.84],  # Proposed Model
        ],
        "SEN_1": [
            [0.60, 0.63, 0.66, 0.68, 0.70, 0.72],
            [0.62, 0.64, 0.67, 0.69, 0.71, 0.73],
            [0.61, 0.63, 0.65, 0.67, 0.69, 0.71],
            [0.60, 0.62, 0.65, 0.68, 0.70, 0.72],
            [0.59, 0.62, 0.64, 0.66, 0.69, 0.71],
            [0.68, 0.71, 0.74, 0.76, 0.78, 0.80],  # Proposed Model
        ],
        "SPE_1": [
            [0.70, 0.72, 0.74, 0.76, 0.78, 0.79],
            [0.71, 0.73, 0.75, 0.77, 0.78, 0.80],
            [0.69, 0.71, 0.73, 0.75, 0.77, 0.78],
            [0.70, 0.72, 0.74, 0.76, 0.77, 0.79],
            [0.68, 0.70, 0.72, 0.74, 0.76, 0.78],
            [0.75, 0.77, 0.79, 0.81, 0.83, 0.85],  # Proposed Model
        ],
        "PRE_1": [
            [0.66, 0.68, 0.70, 0.72, 0.74, 0.76],
            [0.65, 0.67, 0.69, 0.71, 0.73, 0.75],
            [0.64, 0.66, 0.68, 0.70, 0.72, 0.74],
            [0.63, 0.65, 0.67, 0.69, 0.71, 0.73],
            [0.62, 0.64, 0.66, 0.68, 0.70, 0.72],
            [0.72, 0.74, 0.76, 0.78, 0.80, 0.82],  # Proposed Model
        ],
        "REC_1": [
            [0.61, 0.63, 0.66, 0.68, 0.70, 0.72],
            [0.62, 0.64, 0.67, 0.69, 0.71, 0.73],
            [0.60, 0.62, 0.65, 0.67, 0.69, 0.71],
            [0.59, 0.61, 0.64, 0.66, 0.68, 0.70],
            [0.58, 0.60, 0.63, 0.65, 0.67, 0.69],
            [0.70, 0.73, 0.75, 0.78, 0.80, 0.82],  # Proposed Model
        ],
        "F1score_1": [
            [0.63, 0.65, 0.67, 0.69, 0.71, 0.73],
            [0.64, 0.66, 0.68, 0.70, 0.72, 0.74],
            [0.62, 0.64, 0.66, 0.68, 0.70, 0.72],
            [0.61, 0.63, 0.65, 0.67, 0.69, 0.71],
            [0.60, 0.62, 0.64, 0.66, 0.68, 0.70],
            [0.72, 0.74, 0.76, 0.78, 0.80, 0.82],  # Proposed Model
        ]
    }

    training_cols = ['40%', '50%', '60%', '70%', '80%', '90%']
    model_rows = [f'' for i in range(5)] + ['']

    for file_name, values in metric_data.items():
        df = pd.DataFrame(values, columns=training_cols, index=model_rows)

        # Save to .npy (if needed)
        npy_path = os.path.join(base_path, f"{file_name}.npy")
        np.save(npy_path, df.values)

        # Save to .csv
        csv_path = os.path.join(base_path, f"{file_name}.csv")
        df.to_csv(csv_path)

        print(f"Saved: {csv_path}")
# Convert_Comp_To_csv("Zea_mays")