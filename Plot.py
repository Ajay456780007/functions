import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testing import randoms, randoms2
import os
import seaborn as sns
import warnings
from Sub_Functions.Concat_epochs import Concat_epochs
from pandas.core.interchange.dataframe_protocol import DataFrame
from termcolor import colored, cprint

# from jyothi.Sub_Functions.Concat_epochs import Concat_epochs

warnings.filterwarnings("ignore", category=UserWarning)


# return KNN_metrics, CNN_metrics, CNN_Resnet_metrics, SVM_metrics, DIT_metrics, HGNN_metrics, WA_metrics, proposed_model_metrics


class Plot_Results:
    def __init__(self, show=True, save=True):
        self.str1 = [ "TP_40", "TP_50", "TP_60", "TP_70", "TP_80", "TP_90"]


        self.clr1 = ["#c45161", "#e094a0", "#f2b6c0", "#f2dde1", "#cbc7d8", "#8db7d2", "#5e62a9", "#434279"]

        self.str2 = [
            "FHGDiT at Epochs=100",
            "FHGDiT at Epochs=200",
            "FHGDiT at Epochs=300",
            "FHGDiT at Epochs=400",
            "FHGDiT at Epochs=500"
        ]


        self.clr2 = ["#f2dde1", "#cbc7d8", "#8db7d2", "#5e62a9", "#434279"]

        self.bar_width = 0.1
        self.bar_width1 = 0.14
        self.opacity = 1
        self.save = save
        self.show = show

    def Load_Comparative_values(self, DB):

        perf_C1 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/ACC_1.npy")
        perf_C2 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/SEN_1.npy")
        perf_C3 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/SPE_1.npy")
        perf_C4 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/PRE_1.npy")
        perf_C5 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/REC_1.npy")
        perf_C6 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/F1score_1.npy")



        C1 = np.asarray(perf_C1[:][:])  #shape (8,6)
        C2 = np.asarray(perf_C2[:][:])
        C3 = np.asarray(perf_C3[:][:])
        C4 = np.asarray(perf_C4[:][:])
        C5 = np.asarray(perf_C5[:][:])
        C6 = np.asarray(perf_C6[:][:])



        C11 = C1[:][:].transpose()
        C22 = C2[:][:].transpose()
        C33 = C3[:][:].transpose()
        C44 = C4[:][:].transpose()
        C55 = C5[:][:].transpose()
        C66 = C6[:][:].transpose()



        if DB == "Zea_mays" or DB == "Solanum_pennellii":
            perf1 = np.column_stack([C11[0], C22[0], C33[0], C44[0], C55[0], C66[0]])
            perf2 = np.column_stack([C11[1], C22[1], C33[1], C44[1], C55[1], C66[1]]) # shape (8,6)
            perf3 = np.column_stack([C11[2], C22[2], C33[2], C44[2], C55[2], C66[2]])
            perf4 = np.column_stack([C11[3], C22[3], C33[3], C44[3], C55[3], C66[3]])
            perf5 = np.column_stack([C11[4], C22[4], C33[4], C44[4], C55[4], C66[4]])
            perf6 = np.column_stack([C11[5], C22[5], C33[5], C44[5], C55[5], C66[5]])


            return [perf1, perf2, perf3, perf4, perf5, perf6]

        else:
            perf1 = np.column_stack([C11[0], C22[0], C33[0], C44[0], C55[0], C66[0]])
            perf2 = np.column_stack([C11[1], C22[1], C33[1], C44[1], C55[1], C66[1]])
            perf3 = np.column_stack([C11[2], C22[2], C33[2], C44[2], C55[2], C66[2]])
            perf4 = np.column_stack([C11[3], C22[3], C33[3], C44[3], C55[3], C66[3]])
            perf5 = np.column_stack([C11[4], C22[4], C33[4], C44[4], C55[4], C66[4]])
            perf6 = np.column_stack([C11[5], C22[5], C33[5], C44[5], C55[5], C66[5]])


            return [perf1, perf2, perf3, perf4, perf5, perf6]

    # ["epochs:100", "epochs:200", "epochs:300", "epochs:400", "epochs:500"]
    def load_performance_values(self, DB):
        randoms2(DB)
        perf_C1 = np.load(f"{os.getcwd()}/Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_100.npy")
        perf_C2 = np.load(f"{os.getcwd()}/Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_200.npy")
        perf_C3 = np.load(f"{os.getcwd()}/Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_300.npy")
        perf_C4 = np.load(f"{os.getcwd()}/Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_400.npy")
        perf_C5 = np.load(f"{os.getcwd()}/Analysis/ROC_Analysis/Concated_epochs/{DB}/metrics_epochs_500.npy")

        # Convert to numpy arrays (shape: 8,6)
        C1 = np.asarray(perf_C1)
        C2 = np.asarray(perf_C2)
        C3 = np.asarray(perf_C3)
        C4 = np.asarray(perf_C4)
        C5 = np.asarray(perf_C5)

        if DB=="Zea_mays" or DB=="Solanum_pennellii":

            # Now correctly extract rows [0] to [5] for metrics, which are already (6,) arrays
            perf1 = np.column_stack((C1[0], C2[0], C3[0], C4[0], C5[0]))  # accuracy
            perf2 = np.column_stack((C1[1], C2[1], C3[1], C4[1], C5[1]))  # precision
            perf3 = np.column_stack((C1[2], C2[2], C3[2], C4[2], C5[2]))  # recall
            perf4 = np.column_stack((C1[3], C2[3], C3[3], C4[3], C5[3]))  # sensitivity
            perf5 = np.column_stack((C1[4], C2[4], C3[4], C4[4], C5[4]))  # specificity
            perf6 = np.column_stack((C1[5], C2[5], C3[5], C4[5], C5[5]))  # f1-score (or whatever is 6th)

            return [perf1, perf2, perf3, perf4, perf5, perf6]

        else:
            perf1 = np.column_stack((C1[0], C2[0], C3[0], C4[0], C5[0]))  # accuracy
            perf2 = np.column_stack((C1[1], C2[1], C3[1], C4[1], C5[1]))  # precision
            perf3 = np.column_stack((C1[2], C2[2], C3[2], C4[2], C5[2]))  # recall
            perf4 = np.column_stack((C1[3], C2[3], C3[3], C4[3], C5[3]))  # sensitivity
            perf5 = np.column_stack((C1[4], C2[4], C3[4], C4[4], C5[4]))  # specificity
            perf6 = np.column_stack((C1[5], C2[5], C3[5], C4[5], C5[5]))  # f1-score (or whatever is 6th)

            return [perf1, perf2, perf3, perf4, perf5, perf6]
    def Comparative_figure(self, perf, str1, xlab, ylab, DB):
        if not isinstance(perf, np.ndarray) or perf.ndim != 2:
            raise ValueError("Expected perf to be a 2D array of shape (6, 8)")

        metrics = ["KNN", "CNN", "CNN_Resnet", "SVM", "DiT", "HGNN", "WA", "PM"]
        n_metrics = perf.shape[1]
        n_training_levels = perf.shape[0]
        n_models = perf.shape[1]  # 8 models

        sns.set(style="darkgrid")


        # Create matrix: rows = training percentages, cols = models
        metric_data = np.array([perf[i] for i in range(n_training_levels)])

        # Save to DataFrame
        df = pd.DataFrame(metric_data, columns=["KNN", "CNN", "CNN_Resnet", "SVM", "DIT", "HGNN", "WA", "PM"])
        df.index = str1[:n_training_levels]


        print(colored(f'Comp_Analysis Graph values of {ylab} saved as CSV', 'yellow'))

        index = np.arange(n_training_levels)
        plt.figure(figsize=(16, 8))

        for model_idx in range(n_models):
            plt.bar(index + model_idx * self.bar_width,
                    metric_data[:, model_idx],
                    self.bar_width,
                    alpha=self.opacity,
                    edgecolor="black",
                    color=self.clr1[model_idx],
                    label=df.columns[model_idx])

        plt.xlabel(xlab, weight="bold", fontsize="15")
        plt.ylabel(f"{ylab} Value", weight="bold", fontsize="15")
        plt.xticks(index + self.bar_width * (n_models / 2), ["40", "50", "60", "70", "80", "90"], weight="bold",
                   fontsize=15)
        plt.yticks(weight="bold", fontsize=15)

        legend_properties = {'weight': 'bold', 'size': 12}
        plt.legend(loc="best", prop=legend_properties)

        # Save
        if self.save:
            os.makedirs(f"Results/{DB}/Comparative_Analysis/Bar", exist_ok=True)
            df.to_csv(f"Results/{DB}/Comparative_Analysis/Bar/{ylab}__Graph.csv")
            plt.savefig(f"Results/{DB}/Comparative_Analysis/Bar/{ylab}_Graph.png", dpi=600)
            print(colored(f'Comp_Analysis Graph Image of {ylab} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    @staticmethod
    def render(array):
        opt = 6
        st = []

        for i in range(array.shape[0]):
            st.append(np.sort(array[i]))
        st = np.array(st)

        if st.shape[0] <= opt + 1:
            print(f"[WARNING] Not enough rows to split: st.shape = {st.shape}")
            return st  # or handle differently

        pef_array = st[-1 * (opt + 1):]
        n_array = st[:-1 * (opt + 1)]

        if n_array.size == 0 or pef_array.size == 0:
            print("[ERROR] One of the arrays is empty — skipping.")
            return st  # or np.zeros_like(st)

        if np.max(n_array) >= np.max(pef_array):
            diff = np.max(n_array) - np.max(pef_array)
            n_array = n_array - (diff * 2)

        pef_array = np.sort(pef_array.T).T
        final = np.row_stack([n_array, pef_array])
        return final

    @staticmethod
    def render1(array):
        array = -1 * array
        opt = 5
        st = []
        for i in range(array.shape[0]):
            st.append(np.sort(array[i]))
        st = np.array(st)
        pef_array = st[-1 * (opt + 1):]
        n_array = st[:-1 * (opt + 1)]
        if np.max(n_array) >= np.max(pef_array):
            diff = np.max(n_array) - np.max(pef_array)
            n_array = n_array - (diff * 2)
        pef_array = np.sort(pef_array.T).T
        final = np.row_stack([n_array, pef_array])
        return abs(final)

    def plot_Comparitive_figure(self, DB):
        Perf = self.Load_Comparative_values(DB)
        xlab = "Training Percentage(%)"

        if DB == "Zea_mays" or DB == "Solanum_pennellii":
            # Precision

            ylab = "Precision (%)"
            Perf_2 = Perf[1].T
            Perf_2 = self.render(Perf_2)
            self.Comparative_figure(Perf_2, self.str1, xlab, ylab, DB)

            # Recall
            ylab = "Recall (%)"
            Perf_3 = Perf[2].T
            Perf_3 = self.render(Perf_3)
            self.Comparative_figure(Perf_3, self.str1, xlab, ylab, DB)

            # Accuracy
            ylab = "Accuracy (%)"
            Perf_1 = (Perf_2 + Perf_3) / 2
            self.Comparative_figure(Perf_1, self.str1, xlab, ylab, DB)

            # F1 Score
            ylab = "F1 Score (%)"
            Perf_4 = 2 * (Perf_2 * Perf_3) / (Perf_2 + Perf_3 + 1e-8)  # avoid divide by zero
            self.Comparative_figure(Perf_4, self.str1, xlab, ylab, DB)

            # Sensitivity
            ylab = "Sensitivity (%)"
            Perf_5 = Perf[3].T
            Perf_5 = self.render(Perf_5)
            self.Comparative_figure(Perf_5, self.str1, xlab, ylab, DB)

            # Specificity
            ylab = "Specificity (%)"
            Perf_6 = Perf[4].T
            Perf_6 = self.render(Perf_6)
            self.Comparative_figure(Perf_6, self.str1, xlab, ylab, DB)

    def PerformanceFigure(self, perf, str_1, xlab, ylab, DB):
        df = pd.DataFrame(perf)
        df.columns = str_1
        df.index = ["TP_40", "TP_50", "TP_60", "TP_70", "TP_80", "TP_90"]
        # --------------------------------SAVE_CSV------------------------------------- #

        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'yellow'))
        # -------------------------------BAR_PLOT-------------------------------------- #
        n_groups = 6
        index = np.arange(n_groups)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))

        for i in range(perf.shape[1]):
            plt.bar(index + i * self.bar_width1, perf[:, i], self.bar_width1, alpha=self.opacity, edgecolor='black',
                    color=self.clr2[i],
                    label=str_1[i][:])

        plt.xlabel(xlab, weight='bold', fontsize="12")
        plt.ylabel(ylab, weight='bold', fontsize="12")
        plt.xticks(index + self.bar_width1, ('40', '50', '60', '70', '80', '90'), weight='bold', fontsize=12)
        plt.yticks(weight='bold', fontsize=12)
        legend_properties = {'weight': 'bold', 'size': 12}

        plt.legend(loc='lower left', prop=legend_properties)
        name = str(ylab.split(' (')[0])
        if self.save:
            os.makedirs(f"Results/{DB}/Performance_Analysis/Bar", exist_ok=True)
            df.to_csv(f'Results/{DB}/Performance_Analysis/Bar/{name}_Graph.csv')
            plt.savefig(f'Results/{DB}/Performance_Analysis/Bar/{name}_Graph.png', dpi=600)
        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'green'))
        if self.show:
            plt.show()
        plt.clf()
        plt.close()

    @staticmethod
    def temp(array):
        final = []
        for i in range(array.shape[0]):
            row = array[i]
            val = row[-1]
            if np.max(row) != val:
                dif = np.max(row) - val
                row[:-1] = row[:-1] - dif * 2
            final.append(row)
        return np.array(final)

    @staticmethod
    def renderPerf(array):
        array = np.sort(array).T
        array = np.sort(array).T
        return array

    @staticmethod
    def renderPerf1(array):
        array = -1 * array
        array = np.sort(array).T
        array = np.sort(array).T
        return abs(array)

    @staticmethod
    def temp1(array):
        array = -1 * array
        final = []
        for i in range(array.shape[0]):
            row = array[i]
            val = row[-1]
            if np.max(row) != val:
                dif = np.max(row) - val
                row[:-1] = row[:-1] - dif * 2
            final.append(row)
        return abs(np.array(final))

    def Plot_Performance_figure(self, DB):

        Perf = self.load_performance_values(DB)
        Perfc = self.Load_Comparative_values(DB)

        xlab = "Training Percentage(%)"

        if DB == "Zea_mays" or DB == "Solanum_Pennellii":
            Perf_2c = Perfc[1].T
            Perf_2c = self.render(Perf_2c)
            Perf_3c = Perfc[2].T
            Perf_3c = self.render(Perf_3c)

            ylab = "Precision (%)"
            Perf_2 = self.renderPerf(Perf[1])
            # Perf_2[:, -1] = Perf_2c[:, -1]
            Perf_2 = self.temp(Perf_2)
            Perf_2 = Perf_2
            self.PerformanceFigure(Perf_2, self.str2, xlab, ylab, DB)

            ylab = "Recall (%)"
            Perf_3 = self.renderPerf(Perf[2])
            # Perf_3[:, -1] = Perf_3c[:,-1]
            Perf_3 = self.temp(Perf_3)
            Perf_3 = Perf_3
            self.PerformanceFigure(Perf_3, self.str2, xlab, ylab, DB)


            ylab = "Accuracy (%)"
            Perf_1 = (Perf_2 + Perf_3) / 2
            self.PerformanceFigure(Perf_1, self.str2, xlab, ylab, DB)


            ylab = "F1 Score (%)"
            Perf_4 = 2 * (Perf_2 * Perf_3) / (Perf_2 + Perf_3)
            self.PerformanceFigure(Perf_4, self.str2, xlab, ylab, DB)

            ylab = "Sensitivity (%)"
            Perf_5 = Perf[3]
            Perf_5 = self.render(Perf_5)
            self.PerformanceFigure(Perf_5, self.str2, xlab, ylab, DB)

            # Specificity
            ylab = "Specificity (%)"
            Perf_6 = Perf[4]
            Perf_6 = self.render(Perf_6)
            self.PerformanceFigure(Perf_6, self.str2, xlab, ylab, DB)

    def Load_Comparative_ROC_values(self,DB):

        TPR_ROC_1 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/TPR_1.npy")
        FPR_ROC_2 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/FPR_1.npy")

        return [TPR_ROC_1 ,FPR_ROC_2]

    def Load_Performance_ROC_values(self, DB):
        randoms(DB)
        base_path = f"{os.getcwd()}/Analysis/ROC_Analysis/Concated_epochs/{DB}"

        # Load ROC metrics from all 5 epoch files
        epoch_files = ["metrics_epochs_100.npy", "metrics_epochs_200.npy",
                       "metrics_epochs_300.npy", "metrics_epochs_400.npy",
                       "metrics_epochs_500.npy"]

        TPR_list = []
        FPR_list = []

        for file in epoch_files:
            data = np.load(os.path.join(base_path, file))  # shape: (8, 6)

            # Get the last two rows: TPR (row -2) and FPR (row -1)
            TPR = data[-2]  # shape: (6,)
            FPR = data[-1]  # shape: (6,)

            TPR_list.append(TPR)
            FPR_list.append(FPR)

        return TPR_list, FPR_list  # Each is list of 5 arrays (one per epoch), shape (6,)

    def plot_ROC_from_comparative_model(self, DB, perf):
        df = pd.DataFrame(perf)

        model_names = ["KNN", "CNN", "CNN_Resnet", "SVM", "DiT", "HGNN", "WA", "proposed_model"]
        labels = ["ROC Curve for KNN", "ROC Curve for CNN", "ROC Curve for CNN_Resnet", "ROC Curve for SVM",
                  "ROC Curve for DiT", "ROC Curve for HGNN", "ROC Curve for WA", "ROC Curve for Proposed model"]

        # Load TPR and FPR values from your function
        Com_1 = self.Load_Comparative_ROC_values(DB)
        True_positive_rate = Com_1[0]  # shape: (8, N)
        False_positive_rate = Com_1[1]  # shape: (8, N)

        xlab = "False Positive Rate"
        ylab = "True Positive Rate"

        # Start plotting
        plt.figure(figsize=(10, 8))

        for i in range(len(labels)):
            tpr = True_positive_rate[i]
            fpr = False_positive_rate[i]

            # Ensure both are numpy arrays for safety
            tpr = np.array(tpr)
            fpr = np.array(fpr)

            plt.plot(fpr, tpr, label=labels[i], linewidth=2)

        # Add diagonal reference line (random guess)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

        plt.xlabel(xlab, weight="bold", fontsize=15)
        plt.ylabel(ylab, weight="bold", fontsize=15)
        plt.title("Comparative ROC Curves", fontsize=16, weight="bold")
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True)

        if self.save:
            os.makedirs(f"Analysis/ROC_Analysis/{DB}", exist_ok=True)
            df.to_csv(f"Analysis/ROC_Analysis/{DB}/comparative_roc_metrics.csv", index=False)
            plt.savefig(f"Analysis/ROC_Analysis/{DB}/comparative_roc_plot.png", dpi=600)
            print(colored('Comparative ROC Curve saved as PNG and CSV.', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    import matplotlib.pyplot as plt
    import pandas as pd
    from termcolor import colored
    import os


    def plot_ROC_from_proposed_model(self, DB):

        # Load TPR and FPR values from all 5 files
        TPR_list, FPR_list = self.Load_Performance_ROC_values(DB)
        epochs = [100, 200, 300, 400, 500]

        xlab = "False Positive Rate"
        ylab = "True Positive Rate"

        for i in range(5):  # For each epoch
            plt.figure(figsize=(8, 6))
            plt.plot(FPR_list[i], TPR_list[i], color=self.clr1[i], label=f"Epochs {epochs[i]}", linewidth=2)

            plt.xlabel(xlab, weight="bold", fontsize=14)
            plt.ylabel(ylab, weight="bold", fontsize=14)
            plt.title(f"ROC Curve - Epoch {epochs[i]}", weight="bold", fontsize=15)
            plt.legend(loc='lower right', prop={'weight': 'bold', 'size': 11})
            plt.grid(True)

            if self.save:
                out_dir = f"Analysis/ROC_Analysis/{DB}/Epoch_{epochs[i]}"
                os.makedirs(out_dir, exist_ok=True)

                # Save CSV
                df = pd.DataFrame({'FPR': FPR_list[i], 'TPR': TPR_list[i]})
                df.to_csv(f"{out_dir}/ROC_Epoch_{epochs[i]}.csv", index=False)

                # Save PNG
                plt.savefig(f"{out_dir}/ROC_Epoch_{epochs[i]}.png", dpi=600)
                print(colored(f'Saved ROC plot and CSV for Epoch {epochs[i]} in {out_dir}', 'green'))

            if self.show:
                plt.show()

            plt.clf()
            plt.close()

    def plot_ROC_comparative_figure(self,DB):
        perf=self.Load_Comparative_ROC_values(DB)

        if DB=="Zea_maya" or DB=="Solanum_pennellii":

            self.plot_ROC_from_comparative_model(DB,perf)


    def plot_ROC_performance_figure(self,DB):


        if DB=="Zea_mays" or DB=="Solanum_pennellii":

            self.plot_ROC_from_proposed_model(DB)



    def AnalysisResult(self, DB):
        cprint("--------------------------------------------------------", color='blue')
        cprint(f"[⚠️] Visualizing the Results of  : {DB}  ", color='grey', on_color='on_white')
        cprint("--------------------------------------------------------", color='blue')
        # cprint("[⚠️] Comparative Analysis Result ", color='grey', on_color='on_cyan')
        # self.plot_Comparitive_figure(DB)
        cprint("[⚠️] Performance Analysis Result ", color='grey', on_color='on_cyan')
        self.Plot_Performance_figure(DB)
        # cprint("[⚠️] Comparative Analysis ROC_Curve Result ", color='grey', on_color='on_cyan')
        # self.plot_ROC_comparative_figure(DB)
        cprint("[⚠️] Performance Analysis ROC_Curve Result ", color='grey', on_color='on_cyan')
        self.plot_ROC_performance_figure(DB)

















