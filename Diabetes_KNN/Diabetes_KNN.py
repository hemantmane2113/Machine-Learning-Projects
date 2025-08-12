##############################################################################################
# Required Python Packages
##############################################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import joblib

from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,roc_auc_score,auc

from sklearn.neighbors import KNeighborsClassifier


############################################################################################
# File Paths
############################################################################################
FILE_PATH = "datasets/diabetes.csv"
ALL_MODELS_DIR = "Models"
BEST_MODEL_DIR = os.path.join(ALL_MODELS_DIR, "Best_Model")
RANDOM_STATE = 42
TEST_SIZE  = 0.2
k = 20

############################################################################################
# Function name : dataset_statistics
# Description   : Display the statistics
# Author        : Hemant Dattaji Mane
# Date          : 09/08/2025
############################################################################################
def dataset_statistics(df):
    """Print basic statistics"""

    print(df.describe(include = "all"))

############################################################################################
# Function name : handle_missing_values
# Description   : Here '0's are the missing value,will replace it by of the column
# Input         : Original Dataframe
# Output        : Updated Dataframe
# Author        : Hemant Dattaji Mane
# Date          : 10/08/2025
############################################################################################
def handle_missing_values(df):

    print((df == 0).sum())
    print((df==0).sum().sum())

    print(df.isnull().sum())

    feature_only_df = df.drop(columns = "Outcome")
    print(feature_only_df)

    feature_only_df = feature_only_df.apply(lambda x:x.replace(0,x[x != 0].mean()))

    #df["Outcome"] = df["Outcome"]

    df.update(feature_only_df)

    print((feature_only_df == 0).sum())

    df_final = pd.concat([feature_only_df,df["Outcome"]],axis = 1)

    return df_final

############################################################################################
# Function name : split_dataset
# Description   : Splits the dataset into training and testing sets
# Input         : Updated Dataframe
# Output        : 4 splits of the dataset
# Author        : Hemant Dattaji Mane
# Date          : 11/08/2025
############################################################################################

def split_dataset(X,Y,test_size,random_state ):

    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = test_size,random_state = random_state)

    return train_x,test_x,train_y,test_y
    
############################################################################################
# Function name : build_pipeline
# Description   : Build a Pipeline
# Author        : Hemant Dattaji Mane
# Date          : 11/08/2025
############################################################################################
def build_pipeline(k):

    ## object pipe pf Pipeline class gets created
    pipe = Pipeline([
        ("scalar",StandardScaler()),
        ("knn",KNeighborsClassifier(n_neighbors=k))
    ])

    """
    FLOW SUMMARY for build_pipeline()

    1.Inside build_pipeline(k_value)

        *A new Pipeline object is created and stored in the local variable pipe.

            *Step 1 in the pipeline: StandardScaler() (will scale numerical features)

            *Step 2 in the pipeline: KNeighborsClassifier(n_neighbors=k_value) (KNN model with given k_value)

        *pipe is returned.


    2.Object identity

    At this stage, pipe is just a container that knows which preprocessing and model steps it will perform,
    but it hasn't been trained yet (no fitted scaler, no fitted KNN).

    """

    return pipe
    
############################################################################################
# Function name : train_pipeline
# Description   : Train a Pipeline
# Author        : Hemant Dattaji Mane
# Date          : 11/08/2025
############################################################################################
    
def train_pipeline(pipeline,X_train,Y_train):#pipeline here is trained

    """ 
    FLOW SUMMARY for train_pipeline

    *pipeline here is a Pipeline object (the same kind that came from build_pipeline()).

    *.fit(X_train, Y_train) trains the pipeline:

        1.First, it runs StandardScaler().fit_transform() on your training data.

        2.Then it runs KNeighborsClassifier(n_neighbors=k_value).fit() on the scaled data.

    *After .fit() runs, the same pipeline object now holds:

        *The trained scaler (with mean and std saved)

        *The trained KNN model (with the stored training data for distance calculations)

    *The function returns that exact same pipeline object — now trained.

    """
    pipeline.fit(X_train,Y_train)
    
    return pipeline # this pipeline object is now trained

    """
    trained_model(main) and pipeline now point to the same trained pipeline object in memory.

    They are literally the same object — we just gave it another variable name.
    """

############################################################################################
# Function name : save_model
# Dehscription  : Save the model
# Author        : Hemant Dattaji Mane
# Date          : 11/08/2025
############################################################################################
def save_model(model,path):
    
    """ Create directory if it doesn't exist"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    """ save the model to disk"""
    joblib.dump(model,path)
    print(f"Model saved to {path}")


    """
    FLOW SUMMARY for save_model()

    1.Inside save_model(model, path=MODEL_PATH)

        Takes in a trained model object (model) and a file path (path).

        Uses joblib.dump(model, path) to serialize (convert to a byte stream) and store the model object on disk.

        Prints a confirmation message showing where the model has been saved.

        
    2.The function doesn’t return anything — it just saves the file and prints a message.

    3.Object identity

    The object saved to disk is the same object as trained_model in memory at the time of the call.

    We can later load it back into Python using:

    loaded_model = joblib.load(model_filename)
    
    """

############################################################################################
# Function name : Plotting k_vs_accuracy Graph
# Description   : The visual representation of best value of k
# Author        : Hemant Dattaji Mane
# Date          : 11/08/2025
############################################################################################
def k_vs_accuracy_graph(K_List,Accuracy_List):

    plt.plot(K_List,Accuracy_List,marker = 'o')
    plt.xticks(range(min(K_List), max(K_List) + 1))
    plt.title("K vs Accuracy graph")
    plt.xlabel(" Values of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

############################################################################################
# Function name : Plotting Confusion Matrix
# Description   : The confusion Matrix for the best model is displayed
# Author        : Hemant Dattaji Mane
# Date          : 11/08/2025
############################################################################################

def ConfusionMatrix(Conf_Mat,label,best_k):

    disp = ConfusionMatrixDisplay(confusion_matrix= Conf_Mat,display_labels = label)
    disp.plot(cmap = plt.cm.Blues)
    plt.title(f"Conusion Matrix for best k = {best_k}")
    
############################################################################################
# Function name : Plotting ROC-AUC curve
# Description   : The roc-auc curve for best KNN model is plotted
# Author        : Hemant Dattaji Mane
# Date          : 11/08/2025
############################################################################################
def ROC_AUC_curve(fpr,tpr,roc_auc,label): 

    plt.figure(figsize=(8,6))
    plt.plot(fpr,tpr,label = "ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0,1],[0,1],'k--',label = 'Random Guess')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False positive Rate")
    plt.ylabel("True positive Rate")
    plt.title("ROC curve for Diabetes prediction")
    plt.legend()
    plt.show()

############################################################################################
# Function Name : main
# Description   : Main function from where the execution starts
# Author        : Hemant Dattaji Mane
# Date          : 11/08/2025
############################################################################################

def main():

    # 1) Load CSV

    dataset = pd.read_csv(FILE_PATH)

    # 2) Basic Stats

    dataset_statistics(dataset)

    # 3) Handle '0's
    df = handle_missing_values(dataset)

    # 4)split
    X = df.drop(columns = "Outcome")

    Y = df["Outcome"]
    labels = df["Outcome"].unique()

    train_x,test_x,train_y,test_y = split_dataset(X,Y,TEST_SIZE,RANDOM_STATE)

    print("Train_X shape:: ", train_x.shape)
    print("Test_X shape:: ", test_x.shape)
    print("Train_Y shape:: ", train_y.shape)
    print("Test_Y shape:: ", test_y.shape)

    # 5)Build + train pipeline

    best_accuracy = 0
    best_conf_mat = 0
    best_model = None
    best_k = None
    best_fpr = 0
    best_tpr = 0
    best_roc_auc = 0

    Accuracy_List = []
    K_List = []
    

    for k_value in range(1,k+1):
        print(f"\n Training model for k = {k_value}...")
        K_List.append(k_value)
    
        pipeline = build_pipeline(k_value)##pipeline too is an object
        """
        the newly created untrained Pipeline object(ie.pipe) is returned from the function and stored in pipeline.

        At this stage, pipeline is just a container that knows which preprocessing and model steps it will perform,
        but it hasn't been trained yet (no fitted scaler, no fitted KNN).
        """

        trained_model = train_pipeline(pipeline,train_x,train_y)##pipeline object here is input

        """
        The variable pipeline (already built but not trained) is passed into train_pipeline().

        Inside the function, .fit() modifies it in place.

        That modified (trained) object is returned and stored in trained_model.

    
        trained_model and pipeline(returned) now point to the same trained pipeline object in memory.

        They are literally the same object — we just gave it another variable name.
    

        """


        """
        
        FLOW SUMMARY for(buid_pipeline()(create model object) + train_pipeline()(train model object))

        1.Inside build_pipeline()

            pipe (new Pipeline object) → returned → assigned to pipeline

        2.Inside train_pipeline()

            pipeline (passed in) → trained in place → returned → assigned to trained_model

        3.Object identity

            pipe → pipeline → trained_model are just different names for the same object, but at different stages (before and after training).
        
        """
    # 6) Evaluate the model

        y_pred = trained_model.predict(test_x)

        accuracy = accuracy_score(test_y,y_pred)
        Accuracy_List.append(accuracy)

        cls_rpt = classification_report(test_y,y_pred)

        Con_Mat = confusion_matrix(test_y,y_pred)

        y_score = trained_model.predict_proba(test_x)[:, 1] #probabilites for class 1(positive)

        fpr,tpr,thresholds = roc_curve(test_y,y_score)

        roc_auc = auc(fpr,tpr) #Area under ROC-AUC curve

        print(f"Accuracy for k = {k_value}: {accuracy:.4f}")

        # Save every model in ALL_MODELS_DIR
        model_filename = os.path.join(ALL_MODELS_DIR, f"db_knn_k{k_value}.joblib")
        save_model(trained_model, model_filename)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_cls_rpt = cls_rpt
            best_conf_mat = Con_Mat
            best_model = trained_model
            best_k = k_value
            best_fpr = fpr
            best_tpr = tpr
            best_roc_auc  = roc_auc
            best_model_filename = f"db_knn_k_best{k_value}.joblib"


    # 7) Save the model(only the best)

    if best_model:  
        best_model_path = os.path.join(BEST_MODEL_DIR, best_model_filename)
        save_model(best_model, best_model_path)
  
        print(f"Best model saved with k = {best_k} with Accuracy {best_accuracy}")
        print(f"Classification report for the best model: \n",best_cls_rpt)


        """
        trained_model is passed as model (this must already be trained).

        model_filename is passed as path.
        
        """
    
    #8) Plotting the results

    k_vs_accuracy_graph(K_List,Accuracy_List)
    
    ConfusionMatrix(best_conf_mat,labels,best_k)#(for best model only)

    ROC_AUC_curve(fpr,tpr,roc_auc,labels)#(for best model only)

    #9) Sample_Prediction

    

    best_model = joblib.load(best_model_path)

    sample = test_x.iloc[[0]]# 1-row dataframe
    pred = best_model.predict(sample)
    print("Prediction:",pred[0])# 0 = no diabetes, 1 = diabetes


#############################################################################################
# Application Starter
#############################################################################################
if __name__ == "__main__":
    main()

