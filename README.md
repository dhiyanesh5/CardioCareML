# CardioCareML


# Heart Disease Detection using Machine Learning  

## Overview  
This project focuses on predicting heart disease using machine learning models. It utilizes a dataset collected from [Kaggle](https://www.kaggle.com) and evaluates the performance of four different machine learning models: KNN, Support Vector Classifier (SVC), Random Forest, and Logistic Regression.  

## Table of Contents  
1. [Dataset](#dataset)  
2. [Models Used](#models-used)  
3. [Results](#results)  
4. [Requirements](#requirements)  
5. [How to Run](#how-to-run)  
6. [Conclusion](#conclusion)  

---

## Dataset  
The dataset used for this project is sourced from Kaggle. It contains features that help in predicting whether a person has heart disease.  

### Features (Example):  
- Age  
- Gender  
- Chest Pain Type  
- Blood Pressure  
- Cholesterol  
- Maximum Heart Rate Achieved  
- Other relevant features  

---

## Models Used  
The following machine learning models were implemented and evaluated:  
1. **K-Nearest Neighbors (KNN)**  
2. **Support Vector Classifier (SVC)**  
3. **Random Forest**  
4. **Logistic Regression**  

---

## Results  
The performance of each model is summarized below:  

| **Model**           | **Accuracy** | **Precision** | **Recall** | **F1 Score** |  
|----------------------|-------------|--------------|-----------|-------------|  
| *KNN*               | 75.40%      | 0.67         | 0.85      | 0.75        |  
| *SVC*               | 81.96%      | 0.75         | 0.88      | 0.81        |  
| *Random Forest*     | **85.24%**  | **0.80**     | **0.88**  | **0.84**    |  
| *Logistic Regression* | 81.96%    | 0.75         | 0.88      | 0.81        |  

### Key Observations:  
- The **Random Forest** model achieved the highest accuracy of **85.24%** and the best overall F1 score of **0.84**.  
- KNN underperformed compared to other models, achieving an accuracy of **75.40%**.  

---

## Requirements  
To run this project, the following libraries are required:  
- Python 3.x  
- pandas  
- scikit-learn  
- numpy  
- matplotlib  

Install the dependencies using the following command:  
```bash  
pip install pandas scikit-learn numpy matplotlib  
```  

---

## How to Run  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/heart-disease-detection.git  
   cd heart-disease-detection  
   ```  
2. Run the Jupyter Notebook or Python script:  
   ```bash  
   jupyter notebook  
   ```  
3. Execute the steps in the notebook to train and evaluate the models.  

---

## Conclusion  
The Random Forest model outperformed all other models in terms of accuracy and overall performance. This project demonstrates the potential of machine learning in predicting heart disease effectively using clinical and demographic features.  

---

## Future Work  
- Experiment with more advanced models (e.g., XGBoost, LightGBM).  
- Perform hyperparameter tuning to further optimize model performance.  
- Integrate feature engineering techniques to improve predictions.  

---

## Acknowledgments  
- **Dataset Source**: Kaggle  
- Thanks to scikit-learn for providing the machine learning tools.  

