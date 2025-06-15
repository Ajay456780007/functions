from Comparitive_models.KNN import KNN1
from Comparitive_models.CNN import CNN
from Comparitive_models.DiT import Diffusion_transformer
from Comparitive_models.SVM import SVM
from Comparitive_models.CNN_Resnet import CNN_Resnet
from Comparitive_models.proposed_model import proposed_model
from Comparitive_models.HGNN import HGNN
from Comparitive_models.Without_Attention import Without_Attention
import numpy as np
from sklearn.model_selection import train_test_split
from Sub_Functions import Evaluate

def load_data(data): # Function used to load the data

    print("Data Loading Starts...........") #This is the indication that the data loading is started successfully

    feat = np.load(f"data_loader/{data}_Features.npy")  # Loads the encoded dna sequences in variable feat
    label = np.load(f"data_loader/{data}_labels.npy")  # Loads the categorical label in the variable label
    return feat,label


def train_test_split1(data,percent):
    feat = np.load(f"data_loader/{data}_Features.npy")  # Loads the encoded dna sequences in variable feat
    label = np.load(f"data_loader/{data}_labels.npy")  # Loads the categorical label in the variable label
    # We are doing the below steps to take equal amount of data from all the classes
    class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
    class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
    class_2_indices = np.where(label == 2)[0]  # this line chooses all the 2-label index in the label data

    np.random.seed(42)
    selected_class_0 = np.random.choice(class_0_indices, 50,
                                        replace=False)  # randomly chooses 1000 indices from the class_0_indices
    selected_class_1 = np.random.choice(class_1_indices, 50,
                                        replace=False)  # randomly chooses 1000 indices from the class_1_indices
    selected_class_2 = np.random.choice(class_2_indices, 50,
                                        replace=False)  # randomly chooses 1000 indices from the class_2_indices

    selected_indices = np.concatenate(
        [selected_class_0, selected_class_1, selected_class_2])  # joining all the classes together
    np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

    balanced_feat = feat[selected_indices]  # getting the original data from the feat
    balanced_label = label[selected_indices]  # getting the original data from the labels
    data_size = balanced_feat.shape[0]  # Checks the shape of balanced_feat to convert the training_percentage to integer
    actual_percentage = int((data_size / 100) * percent)  # Converted the float training percentage to integer
    training_sequence = balanced_feat[:actual_percentage]  # splitting the training data
    training_labels = balanced_label[:actual_percentage]  # splitting the training label
    testing_sequence = balanced_feat[actual_percentage:]   # splitting the testing sequence
    testing_labels = balanced_label[actual_percentage:]     # splitting the Testing labels

    return training_sequence,testing_sequence,training_labels,testing_labels   #The function  train_test_split1 return the training and testing data



def train_test_split2(data,percent):
    feat = np.load(f"data_loader/{data}_Features.npy")  # Loads the encoded dna sequences in variable feat
    label = np.load(f"data_loader/{data}_labels.npy")  # Loads the categorical label in the variable label
    # We are doing the below steps to take equal amount of data from all the classes
    class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
    class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
    class_2_indices = np.where(label == 2)[0]  # this line chooses all the 2-label index in the label data

    np.random.seed(42)
    selected_class_0 = np.random.choice(class_0_indices, 4,
                                        replace=False)  # randomly chooses 1000 indices from the class_0_indices
    selected_class_1 = np.random.choice(class_1_indices, 4,
                                        replace=False)  # randomly chooses 1000 indices from the class_1_indices
    selected_class_2 = np.random.choice(class_2_indices, 4,
                                        replace=False)  # randomly chooses 1000 indices from the class_2_indices

    selected_indices = np.concatenate(
        [selected_class_0, selected_class_1, selected_class_2])  # joining all the classes together
    np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

    balanced_feat = feat[selected_indices]  # getting the original data from the feat
    balanced_label = label[selected_indices]  # getting the original data from the labels
    data_size = balanced_feat.shape[0]  # Checks the shape of balanced_feat to convert the training_percentage to integer
    actual_percentage = int(data_size * percent)  # Converted the float training percentage to integer
    training_sequence = balanced_feat[:actual_percentage]  # splitting the training data
    training_labels = balanced_label[:actual_percentage]  # splitting the training label
    testing_sequence = balanced_feat[actual_percentage:]   # splitting the testing sequence
    testing_labels = balanced_label[actual_percentage:]     # splitting the Testing labels

    return training_sequence,testing_sequence,training_labels,testing_labels   #The function  train_test_split1 return the training and testing data



def models_return_metrics(data,ok=True,percent=None):

    KNN_metrics=[]      # empty list to store KNN metrics
    CNN_metrics=[]      # empty list to store the CNN metrics
    CNN_Resnet_metrics=[]   # empty list to store  the CNN_Resnet metrics
    DIT_metrics=[]          #  empty list to store  the DiT metrics
    HGNN_metrics=[]         # empty list to store  the HGNN metrics
    SVM_metrics=[]          # empty list to store  the SVM metrics
    WA_metrics=[]           # empty list to store  the WA metrics
    proposed_model_metrics=[]   # empty list to store  the proposed_model metrics

    training_percentage = [40, 50, 60, 70, 80, 90]
    if ok:
        for i in training_percentage:
            x_train, x_test, y_train, y_test = train_test_split1(data, i)

            # First Comparative Model KNN
            KNN_metrics1=KNN1(x_train,y_train,x_test,y_test) # This line passes training and testing data to the KNN model(To Visit KNN1 select KNN1 and press ctrl+b(in pycharm)))

          # in the model the metrics are evaluated and returned in a list which is appended to the empty list
            KNN_metrics.append(KNN_metrics1)



            #This is the Second Comparative model CNN
            CNN_metrics1=CNN(x_train,x_test,y_train,y_test) #This line passes the training and testing data to the CNN model
            CNN_metrics.append(CNN_metrics1) # This line appended the returned metrics to the empty list
            print(f"The metrics for KNN at {i}% training data:", CNN_metrics1)

            #This is the third comparative model
            Dit_metrics1=Diffusion_transformer(x_train,x_test,y_train,y_test)  # This line passes train and test to the Diffusion transformer
            DIT_metrics.append(Dit_metrics1) # the returned metrics are  appended to the empty list above

            #This is the fourth Comparative model
            HGNN_metrics1=HGNN(x_train,x_test,y_train,y_test)  # this line passes train and test data to the HGNN model
            HGNN_metrics.append(HGNN_metrics1) # The returned metrics data is appended to the empty list above

            #This is the fifth comparative model
            SVM_metrics1=SVM(x_train,x_test,y_train,y_test) # This line passes the data to the  SVM model
            SVM_metrics.append(SVM_metrics1) # the returned metrics are stored in the empty list above

            #THis is the sixth Comparative model
            WA_metrics1 = Without_Attention(x_train, x_test, y_train, y_test, i) # This line passes the data to the  Proposed model without attention layer model
            WA_metrics.append(WA_metrics1)  # the returned metrics are stored in the empty list above

            # THis is the seventh Comparative model
            CNN_Resnet_metrics1=CNN_Resnet(x_train,x_test,y_train,y_test)  # This line passes the data to the  CNN_Resnet model
            CNN_Resnet_metrics.append(CNN_Resnet_metrics1)  # the returned metrics are stored in the empty list above

            # THis is the proposed  model
            Proposed_model_metrics1=proposed_model(x_train,x_test,y_train,y_test,i)  # This line passes the data to the  Proposed  model
            proposed_model_metrics.append(Proposed_model_metrics1)  # the returned metrics are stored in the empty list above
    else:
        x_train, x_test, y_train, y_test = train_test_split1(data, percent)

        # First Comparative Model KNN
        KNN_metrics1 = KNN1(x_train, y_train, x_test,
                            y_test)  # This line passes training and testing data to the KNN model(To Visit KNN1 select KNN1 and press ctrl+b(in pycharm)))

        # in the model the metrics are evaluated and returned in a list which is appended to the empty list
        KNN_metrics.append(KNN_metrics1)

        # This is the Second Comparative model CNN
        CNN_metrics1 = CNN(x_train, x_test, y_train,
                           y_test)  # This line passes the training and testing data to the CNN model
        CNN_metrics.append(CNN_metrics1)  # This line appended the returned metrics to the empty list
        print(f"The metrics for KNN at {i}% training data:", CNN_metrics1)

        # This is the third comparative model
        Dit_metrics1 = Diffusion_transformer(x_train, x_test, y_train,
                                             y_test)  # This line passes train and test to the Diffusion transformer
        DIT_metrics.append(Dit_metrics1)  # the returned metrics are  appended to the empty list above

        # This is the fourth Comparative model
        HGNN_metrics1 = HGNN(x_train, x_test, y_train, y_test)  # this line passes train and test data to the HGNN model
        HGNN_metrics.append(HGNN_metrics1)  # The returned metrics data is appended to the empty list above

        # This is the fifth comparative model
        SVM_metrics1 = SVM(x_train, x_test, y_train, y_test)  # This line passes the data to the  SVM model
        SVM_metrics.append(SVM_metrics1)  # the returned metrics are stored in the empty list above

        # THis is the sixth Comparative model
        WA_metrics1 = Without_Attention(x_train, x_test, y_train, y_test,
                                        i)  # This line passes the data to the  Proposed model without attention layer model
        WA_metrics.append(WA_metrics1)  # the returned metrics are stored in the empty list above

        # THis is the seventh Comparative model
        CNN_Resnet_metrics1 = CNN_Resnet(x_train, x_test, y_train,
                                         y_test)  # This line passes the data to the  CNN_Resnet model
        CNN_Resnet_metrics.append(CNN_Resnet_metrics1)  # the returned metrics are stored in the empty list above

        # THis is the proposed  model
        Proposed_model_metrics1 = proposed_model(x_train, x_test, y_train, y_test,
                                                 i)  # This line passes the data to the  Proposed  model
        proposed_model_metrics.append(Proposed_model_metrics1)  # the returned metrics are stored in the empty list above

        # This are the metrics each model returns such as Accuracy , Sensitivity, Specificity,Precision,Recall,True positive rate , False positive rate etc..
    return KNN_metrics,CNN_metrics,CNN_Resnet_metrics,SVM_metrics,DIT_metrics,HGNN_metrics,WA_metrics,proposed_model_metrics






