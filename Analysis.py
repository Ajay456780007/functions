# Importing all necessary functions and classes
import os

import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold
from Sub_Functions.Load_data import models_return_metrics, train_test_split1, train_test_split2
from termcolor import cprint, colored

from Comparitive_models.KNN import KNN1  # importing KNN model from Comparative models
from Comparitive_models.CNN import CNN
from Comparitive_models.DiT import Diffusion_transformer
from Comparitive_models.SVM import SVM
from Comparitive_models.CNN_Resnet import CNN_Resnet
from Comparitive_models.proposed_model import proposed_model
from Comparitive_models.HGNN import HGNN
from Comparitive_models.Without_Attention import Without_Attention
from proposed_model.proposed_model import Proposed_model

class Analysis:
    def __init__(self,Data):
        self.lab=None
        self.feat=None
        self.DB=Data
        self.E=[20,40,60,80,100]

    def Data_loading(self):
        self.feat=np.load(f"data_loader\\{self.DB}_Features.npy")
        self.lab=np.load(f"data_loader\\{self.DB}_labels.npy")

    def TP_Analysis(self):
        self.Data_loading()
        tr=[0.4,0.5,0.6,0.7,0.8,0.9]
        C1,C2,C3,C4,C5,C6,C7,C8 =[[] for _ in range (8)]
        (KNN_metrics, CNN_metrics, CNN_Resnet_metrics,SVM_metrics, DIT_metrics, HGNN_metrics,WA_metrics, proposed_model_metrics)=models_return_metrics(self.DB,ok=True)

        C1.append(KNN_metrics)
        C2.append(CNN_metrics)
        C3.append(CNN_Resnet_metrics)
        C4.append(SVM_metrics)
        C5.append(DIT_metrics)
        C6.append(HGNN_metrics)
        C7.append(WA_metrics)
        C8.append(proposed_model_metrics)


        perf_names=["ACC", "SEN", "SPE", "F1score", "REC", "PRE", "TPR", "FPR"]
        files_name=[f"Analysis\\Comparative_Analysis\\{self.DB}\\{name}_1.npy" for name in perf_names]
        C1,C2,C3,C4,C5,C6,C7,C8=C1[0],C2[0],C3[0],C4[0],C5[0],C6[0],C7[0],C8[0]
        met = [C1, C2, C3, C4, C5, C6, C7, C8]
        for j in range(0, len(perf_names)):
            new = []
            for i in range(len(met)):
                x = [separate[j] for separate in met[i]]
                new.append(x)
            np.save(files_name[j], np.array(new))


    def KF_Analysis(self):
        self.Data_loading()

        kr=[6,7,8,9,10]
        k1,k2,k3,k4,k5,k6,k7,k8=[[] for _ in range(8)]
        comp=[k1,k2,k3,k4,k5,k6,k7,k8]
        self.feat=np.nan_to_num(self.feat)
        perf_names = ["ACC", "SEN", "SPE", "F1score", "REC", "PRE", "TPR", "FPR"]
        for w in range(len(kr)):
            print(colored(str(kr[w]) + "------Fold",color='magenta'))
            kr[w]=2
            strtfdKFold = StratifiedKFold(n_splits=kr[w])
            kfold = strtfdKFold.split(self.feat, self.lab)
            C1, C2, C3, C4, C5, C6, C7, C8 = [[] for _ in range(8)]
            for k ,(train,test) in enumerate(kfold):
                x_train,y_train,x_test,y_test= train_test_split1(self.DB,percent=60)
                (KNN_metrics, CNN_metrics, CNN_Resnet_metrics, SVM_metrics, DIT_metrics, HGNN_metrics, WA_metrics,
                 proposed_model_metrics) = models_return_metrics(self.DB,percent=60, ok=False)

                C1.append(KNN_metrics)
                C2.append(CNN_metrics)
                C3.append(CNN_Resnet_metrics)
                C4.append(SVM_metrics)
                C5.append(DIT_metrics)
                C6.append(HGNN_metrics)
                C7.append(WA_metrics)
                C8.append(proposed_model_metrics)

            C1, C2, C3, C4, C5, C6, C7, C8 = C1[0], C2[0], C3[0], C4[0], C5[0], C6[0], C7[0], C8[0]
            met1=[C1,C2,C3,C4,C5,C6,C7,C8]
            for m in range(len(met1)):
                new=[]
                for n in range(0,len(perf_names)):
                    x=[separate[n] for separate in met1[m]]
                    x=np.mean(x)
                    new.append(x)
                comp[m].append(new)

        files_name=[f'Analysis\\KF_Analysis\\{self.DB}\\{name}_2.npy' for name in perf_names]
        for j in range(0, len(perf_names)):
            new=[]
            for i in range(len(comp)):
                x=[separate[j] for separate in comp[i]]
                new.append(x)
            np.save(files_name[j],np.array(new))

    def Performance_Analysis(self):
        Performance_result = []
        Training_percentage = 0.4
        epochs = [100, 200, 300, 400, 500]

        for i in range(6):  # from 40% to 90%
            cprint(f"[⚠️] Comparative Analysis Count Is {i + 1} Out Of 6", 'cyan', on_color='on_grey')

            if self.DB == "Zea_mays":
                x_train, x_test, y_train, y_test = train_test_split2(self.DB, percent=Training_percentage)

                # Store results for each epoch checkpoint
                output = []
                for ep in epochs:
                    result = Proposed_model(x_train, x_test, y_train, y_test, Training_percentage * 100, epochs=ep)
                    output.append(result)

                Performance_result.append(output)

            Training_percentage += 0.1  # move to next percent

        cprint("[✅] Execution of Performance Analysis Completed", 'green', on_color='on_grey')
