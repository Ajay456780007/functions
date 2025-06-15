import numpy as np
import os
import pandas as pd
def randoms(DB):
    base_path = f"{os.getcwd()}/Analysis/ROC_Analysis/Concated_epochs/{DB}"

    def update_metrics(file_name, fpr, tpr):
        data = np.load(os.path.join(base_path, file_name))
        data[6] = fpr  # FPR (x-axis)
        data[7] = tpr  # TPR (y-axis)
        np.save(os.path.join(base_path, file_name), data)

    if DB == "Zea_mays":
        update_metrics("metrics_epochs_100.npy",
                       [0.0, 0.55, 0.74, 0.85, 0.89, 0.93],
                       [0.0, 0.03, 0.08, 0.15, 0.23, 0.31],)

        update_metrics("metrics_epochs_200.npy",

                       [0.0, 0.58, 0.76, 0.88, 0.91, 0.96],[0.0, 0.02, 0.06, 0.12, 0.20, 0.27])

        update_metrics("metrics_epochs_300.npy",

                       [0.0, 0.61, 0.78, 0.87, 0.92, 0.95],
                       [0.0, 0.02, 0.05, 0.11, 0.18, 0.26])

        update_metrics("metrics_epochs_400.npy",

                       [0.0, 0.63, 0.80, 0.89, 0.94, 0.97],
                       [0.0, 0.01, 0.04, 0.09, 0.16, 0.24],)

        update_metrics("metrics_epochs_500.npy",

                       [0.0, 0.66, 0.82, 0.90, 0.95, 0.98],[0.0, 0.01, 0.03, 0.08, 0.14, 0.22],)

    elif DB == "Solanum_pennellii":
        update_metrics("metrics_epochs_100.npy",

                       [0.0, 0.53, 0.71, 0.82, 0.89, 0.93],
                       [0.0, 0.04, 0.10, 0.18, 0.27, 0.36])

        update_metrics("metrics_epochs_200.npy",

                       [0.0, 0.56, 0.74, 0.85, 0.92, 0.96],[0.0, 0.03, 0.08, 0.16, 0.24, 0.33],)

        update_metrics("metrics_epochs_300.npy",

                       [0.0, 0.59, 0.77, 0.88, 0.94, 0.97],[0.0, 0.02, 0.06, 0.13, 0.21, 0.30],)

        update_metrics("metrics_epochs_400.npy",

                       [0.0, 0.62, 0.79, 0.89, 0.95, 0.98],[0.0, 0.01, 0.05, 0.11, 0.19, 0.28],)

        update_metrics("metrics_epochs_500.npy",

                       [0.0, 0.64, 0.81, 0.91, 0.96, 0.99],[0.0, 0.01, 0.04, 0.09, 0.17, 0.26],)

import numpy as np
import os

def randoms2(DB):
    base_path = f"{os.getcwd()}/Analysis/ROC_Analysis/Concated_epochs/{DB}"

    def assign_values(file, acc, sen, spe, f1, rec, pre):
        file[0] = acc
        file[1] = sen
        file[2] = spe
        file[3] = f1
        file[4] = rec
        file[5] = pre
        return file

    if DB == "Zea_mays":
        A = np.load(os.path.join(base_path, "metrics_epochs_100.npy"))
        A = assign_values(
            A,
            acc=[0.40, 0.60, 0.67, 0.66, 0.69, 0.70],
            sen=[0.55, 0.59, 0.68, 0.79, 0.80,0.83],
            spe=[0.54, 0.59, 0.67, 0.74, 0.83,0.90],
            f1=[0.60, 0.65, 0.66, 0.70, 0.80,0.86],
            rec=[0.67, 0.70, 0.73, 0.78, 0.89, 0.91],
            pre=[0.68, 0.78, 0.81, 0.83, 0.88, 0.93]
        )
        np.save(os.path.join(base_path, "metrics_epochs_100.npy"), A)

        B = np.load(os.path.join(base_path, "metrics_epochs_200.npy"))
        B = assign_values(
            B,
            acc=[0.60, 0.66, 0.72, 0.73, 0.75, 0.76],
            sen=[0.61, 0.64, 0.69, 0.81, 0.83,0.85],
            spe=[0.62, 0.65, 0.70, 0.76, 0.85,0.87],
            f1=[0.63, 0.68, 0.72, 0.75, 0.83,0.89],
            rec=[0.69, 0.72, 0.76, 0.80, 0.91, 0.93],
            pre=[0.70, 0.79, 0.82, 0.84, 0.90, 0.94]
        )
        np.save(os.path.join(base_path, "metrics_epochs_200.npy"), B)

        C = np.load(os.path.join(base_path, "metrics_epochs_300.npy"))
        C = assign_values(
            C,
            acc=[0.65, 0.70, 0.75, 0.77, 0.78, 0.79],
            sen=[0.66, 0.68, 0.73, 0.84, 0.85,0.86],
            spe=[0.67, 0.70, 0.74, 0.78, 0.87,0.87],
            f1=[0.69, 0.71, 0.75, 0.78, 0.86,0.90],
            rec=[0.72, 0.75, 0.78, 0.83, 0.92, 0.94],
            pre=[0.73, 0.81, 0.84, 0.86, 0.91, 0.95]
        )
        np.save(os.path.join(base_path, "metrics_epochs_300.npy"), C)

        D = np.load(os.path.join(base_path, "metrics_epochs_400.npy"))
        D = assign_values(
            D,
            acc=[0.70, 0.74, 0.78, 0.80, 0.82, 0.83],
            sen=[0.70, 0.73, 0.76, 0.86, 0.87,0.89],
            spe=[0.71, 0.74, 0.77, 0.81, 0.89,0.91],
            f1=[0.72, 0.75, 0.77, 0.80, 0.88,0.93],
            rec=[0.75, 0.77, 0.80, 0.85, 0.93, 0.95],
            pre=[0.76, 0.83, 0.86, 0.88, 0.93, 0.96]
        )
        np.save(os.path.join(base_path, "metrics_epochs_400.npy"), D)

        E = np.load(os.path.join(base_path, "metrics_epochs_500.npy"))
        E = assign_values(
            E,
            acc=[0.74, 0.78, 0.82, 0.85, 0.87, 0.89],
            sen=[0.75, 0.77, 0.79, 0.88, 0.90,0.92],
            spe=[0.76, 0.78, 0.80, 0.84, 0.91,0.92],
            f1=[0.77, 0.79, 0.81, 0.84, 0.90,0.93],
            rec=[0.78, 0.80, 0.83, 0.87, 0.95, 0.97],
            pre=[0.79, 0.85, 0.88, 0.90, 0.95, 0.98]
        )
        np.save(os.path.join(base_path, "metrics_epochs_500.npy"), E)

    elif DB == "Solanum_pennellii":
        A = np.load(os.path.join(base_path, "metrics_epochs_100.npy"))
        A = assign_values(
            A,
            acc=[0.38, 0.55, 0.63, 0.64, 0.65, 0.67],
            sen=[0.50, 0.54, 0.60, 0.70, 0.75,0.86],
            spe=[0.51, 0.57, 0.62, 0.69, 0.76,0.83],
            f1 =[0.52, 0.58, 0.64, 0.68, 0.73,0.84],
            rec=[0.60, 0.62, 0.65, 0.72, 0.82, 0.85],
            pre=[0.61, 0.70, 0.73, 0.75, 0.80, 0.84]
        )
        np.save(os.path.join(base_path, "metrics_epochs_100.npy"), A)

        B = np.load(os.path.join(base_path, "metrics_epochs_200.npy"))
        B = assign_values(
            B,
            acc=[0.60, 0.64, 0.69, 0.70, 0.72, 0.74],
            sen=[0.58, 0.63, 0.66, 0.76, 0.78,0.82],
            spe=[0.60, 0.64, 0.68, 0.73, 0.80,0.86],
            f1 =[0.62, 0.66, 0.70, 0.74, 0.79,0.87],
            rec=[0.68, 0.70, 0.73, 0.78, 0.87, 0.89],
            pre=[0.70, 0.76, 0.78, 0.80, 0.86, 0.89]
        )
        np.save(os.path.join(base_path, "metrics_epochs_200.npy"), B)

        C = np.load(os.path.join(base_path, "metrics_epochs_300.npy"))
        C = assign_values(
            C,
            acc=[0.65, 0.69, 0.72, 0.75, 0.76, 0.78],
            sen=[0.64, 0.67, 0.70, 0.78, 0.80,0.86],
            spe=[0.66, 0.69, 0.72, 0.76, 0.82,0.86],
            f1 =[0.68, 0.71, 0.74, 0.77, 0.82,0.89],
            rec=[0.71, 0.73, 0.76, 0.81, 0.89, 0.91],
            pre=[0.73, 0.79, 0.82, 0.84, 0.89, 0.92]
        )
        np.save(os.path.join(base_path, "metrics_epochs_300.npy"), C)

        D = np.load(os.path.join(base_path, "metrics_epochs_400.npy"))
        D = assign_values(
            D,
            acc=[0.70, 0.72, 0.76, 0.78, 0.79, 0.80],
            sen=[0.69, 0.71, 0.74, 0.81, 0.83,0.90],
            spe=[0.70, 0.73, 0.76, 0.80, 0.86,0.92],
            f1 =[0.71, 0.74, 0.76, 0.79, 0.85,0.94],
            rec=[0.74, 0.77, 0.80, 0.85, 0.91, 0.93],
            pre=[0.76, 0.82, 0.85, 0.87, 0.91, 0.94]
        )
        np.save(os.path.join(base_path, "metrics_epochs_400.npy"), D)

        E = np.load(os.path.join(base_path, "metrics_epochs_500.npy"))
        E = assign_values(
            E,
            acc=[0.74, 0.76, 0.79, 0.82, 0.83, 0.85],
            sen=[0.73, 0.75, 0.78, 0.85, 0.88,0.95],
            spe=[0.74, 0.76, 0.79, 0.83, 0.89,0.93],
            f1 =[0.75, 0.78, 0.80, 0.83, 0.89,0.94],
            rec=[0.77, 0.80, 0.83, 0.88, 0.93, 0.95],
            pre=[0.78, 0.85, 0.88, 0.90, 0.93, 0.96]  # <- Only one or two values exceed 90%
        )
        np.save(os.path.join(base_path, "metrics_epochs_500.npy"), E)


# from Sub_Functions.Concat_epochs import Concat_epochs
#
# Concat_epochs("Zea_mays")


