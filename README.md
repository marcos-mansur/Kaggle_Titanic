# Titanic survival prediction

This repo contains code to a ML model that predicts survivability (classification task) at the Tatinic sinking tragedy based on data from passengers such as Age, Gender (Sex), how much paid for the ticked etc.

The dataset consists of heterogeneous data types and each type requires specific preprocessing/tratement. The final estimator is a voting classifier with Logistic Regression, Random Forest and Gradient Boosting Classifier.

This model scores top 3% at the ["Titanic - Machine Learning from Disaster" Kaggle competition](https://www.kaggle.com/c/titanic).

## Summary

### Research Enviroment (Notebooks)
- [EDA - Exploratory Data Analysis](https://github.com/marcos-mansur/Kaggle_Titanic/blob/main/EDA.ipynb) - Data analysis to better comprehend the features correlations, distribuitions and general behaviour.
- [Feature Engineering](https://github.com/marcos-mansur/Kaggle_Titanic/blob/main/feature_engineering.ipynb) - Feature engineering functions and initial analysis of each transformation's impact over accuracy.
- [Model Pipeline](https://github.com/marcos-mansur/Kaggle_Titanic/blob/main/best_model.ipynb) -  Notebook with code to the TOP 3% model pipeline, validation scores and learning curve.

### Production Environment (.py files)
- [Pipeline](https://github.com/marcos-mansur/Kaggle_Titanic/blob/main/Pipeline.py) - Pipelines workflow 
- [train](https://github.com/marcos-mansur/Kaggle_Titanic/blob/main/train.py) - code to train the model, make a prediction, save the prediction to HD and activate GitHub Actions, showing train and validation scores of diferent pipelines at pull request (CI).
- [Continuous Integration .yaml file](https://github.com/marcos-mansur/Kaggle_Titanic/blob/main/.github/workflows/cml.yaml) - .yaml file responsible for activating GitHub Actions with "push" as trigger.
- [Streamlit Web App Deploy (WIP)](https://github.com/marcos-mansur/Kaggle_Titanic/blob/main/MyApp.py) - first draft of the web app using streamlit to deploy the model 

### Developers:
- Marcos Mansur
- Thiago Ouverney 
