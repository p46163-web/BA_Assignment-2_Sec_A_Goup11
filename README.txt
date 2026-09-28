HR Attrition project with compact 8-input model.

Features: Age, BusinessTravel, Department, JobRole, MaritalStatus, MonthlyIncome, OverTime, YearsAtCompany
Target: Attrition (Yes=1, No=0).
Split: stratified 80/20, random_state=42.
Model: Logistic Regression with balanced class weights.
Run notebook to regenerate model and reports. For app: keep app.py and attrition_model.sav together; pip install -r requirements.txt; streamlit run app.py.

                       Metric                                                                                            Value
                       Target                                                                                        Attrition
                        Model                                                                     Balanced Logistic Regression
               Input features Age, BusinessTravel, Department, JobRole, MaritalStatus, MonthlyIncome, OverTime, YearsAtCompany
                         Rows                                                                                             1470
Missing cells before cleaning                                                                                                0
 Missing cells after cleaning                                                                                                0
       Duplicate rows removed                                                                                                0
             Train rows (80%)                                                                                             1176
              Test rows (20%)                                                                                              294
                     Accuracy                                                                                         0.731293
            Balanced accuracy                                                                                         0.702257
    Precision (Attrition=Yes)                                                                                         0.329787
       Recall (Attrition=Yes)                                                                                         0.659574
     F1 score (Attrition=Yes)                                                                                         0.439716
                      ROC-AUC                                                                                          0.73822
                          MCC                                                                                         0.317872
                     Log loss                                                                                         0.568411
                True Negative                                                                                              184
               False Positive                                                                                               63
               False Negative                                                                                               16
                True Positive                                                                                               31
                           R²                                                          Not applicable to binary classification