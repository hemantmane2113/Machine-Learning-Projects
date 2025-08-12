          Diabetes Prediction Using KNN and Pipeline                                              
                                                       
This project demonstrates an end-to-end machine learning pipeline for predicting diabetes(positive or negavtive) using the Diabetes dataset

It follows industrial bbest practices by:
	a) Automating processing with Pipelines
	b) Handling missing values(0) using .mean()
	c) Using K-nearest neighbour as a machine learning model
	d) Saving and loading trained models using Joblib
	e) Providing data visualization for interpretability
	
Dataset Information
Source: Kaggle
Features(8):
 	1)Pregnancies
 	2)Glucose
 	3)BloodPressure
 	4)SkinThickness
 	5)Insulin
 	6)BMI
 	7)DiabetesPedigreeFunction
 	8)Age
 Target:
 	1)Outcome:
 	        0 = Negative(No Diabetes)
 	        1 = Positive(Has Diabetes

Workflow:

Data Preparation
	Replace missing values(0) with the mean of that column

Train-Test-Split
	Split into 80% train and 20% test
	
Pipeline Construction
	step1: Standardscalar is used so that all features are on the same scale, ensuring KNN’s distance 	          calculations treat them equally.
	step2: KNN classifier

Model Training and Evaluation
	1)Metrics:Accuracy,Confusion Matrix,Classification Report,ROC-AUC score
	2)Feature Importance Plot:Shows most influential features
	


Model Saving and Loading
	1)Save model with Joblib
	2)Load model for feature predictions without retraining
	
Running the project

Train and Evaluate Model

Diabetes_KNN.py

Training model for k = 1...
Accuracy for k = 1: 0.6688
Model saved to Models/db_knn_k1.joblib

 Training model for k = 2...
Accuracy for k = 2: 0.6299
Model saved to Models/db_knn_k2.joblib

 Training model for k = 3...
Accuracy for k = 3: 0.7078
Model saved to Models/db_knn_k3.joblib

 Training model for k = 4...
Accuracy for k = 4: 0.6883
Model saved to Models/db_knn_k4.joblib

 Training model for k = 5...
Accuracy for k = 5: 0.7208
Model saved to Models/db_knn_k5.joblib

 Training model for k = 6...
Accuracy for k = 6: 0.7078
Model saved to Models/db_knn_k6.joblib

 Training model for k = 7...
Accuracy for k = 7: 0.7143
Model saved to Models/db_knn_k7.joblib

 Training model for k = 8...
Accuracy for k = 8: 0.7273
Model saved to Models/db_knn_k8.joblib


 Training model for k = 9...
Accuracy for k = 9: 0.7013
Model saved to Models/db_knn_k9.joblib

 Training model for k = 10...
Accuracy for k = 10: 0.7532
Model saved to Models/db_knn_k10.joblib

 Training model for k = 11...
Accuracy for k = 11: 0.7403
Model saved to Models/db_knn_k11.joblib

 Training model for k = 12...
Accuracy for k = 12: 0.7403
Model saved to Models/db_knn_k12.joblib

 Training model for k = 13...
Accuracy for k = 13: 0.7403
Model saved to Models/db_knn_k13.joblib

 Training model for k = 14...
Accuracy for k = 14: 0.7468
Model saved to Models/db_knn_k14.joblib

 Training model for k = 15...
Accuracy for k = 15: 0.7662
Model saved to Models/db_knn_k15.joblib

 Training model for k = 16...
Accuracy for k = 16: 0.7597
Model saved to Models/db_knn_k16.joblib

 Training model for k = 17...
Accuracy for k = 17: 0.7597
Model saved to Models/db_knn_k17.joblib

 Training model for k = 18...
Accuracy for k = 18: 0.7727
Model saved to Models/db_knn_k18.joblib



Training model for k = 19...
Accuracy for k = 19: 0.7662
Model saved to Models/db_knn_k19.joblib

Training model for k = 20...
Accuracy for k = 20: 0.7662
Model saved to Models/db_knn_k20.joblib

Model saved to Models/Best_Model/db_knn_k_best18.joblib
Best model saved with k = 18 with Accuracy 0.7727272727272727
Classification report for the best model: 
               precision    recall  f1-score   support

           0       0.80      0.87      0.83        99
           1       0.72      0.60      0.65        55

    accuracy                           0.77       154
   macro avg       0.76      0.73      0.74       154
weighted avg       0.77      0.77      0.77       154

Prediction: 1

Visualization
	Feature Importance
	K vs Accuracy graph
            ROC-AUC curve

Model Storage

All the Models saved in the  Model directory while the best model is saved in Model/Best_Model directory

Sample Prediction
    best_model = joblib.load(best_model_path)

    sample = test_x.iloc[[0]]# 1-row dataframe
    pred = best_model.predict(sample)
    print("Prediction:",pred[0])# 0 = no diabetes, 1 = diabetes
    
 Author
 Hemant Dattaji Mane
 Date:09/08/2025

 		
   
