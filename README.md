# **Employee Retention Analysis in the Healthcare Sector**
## **Project Overview**

This repository contains analysis scripts, datasets, and models related to the Employee Retention of Physicians. The primary goal is to explore factors influencing the retention of medical professionals in the healthcare sector, with an emphasis on physicians.

## **Repository Contents**
- EmployeeRetention.csv: Raw data collected from surveys about employee satisfaction and retention factors.
- EmployeeRetentionData_C.csv: Cleaned and preprocessed data ready for analysis.
- EmployeeRetention of Physicians.py: Python script for statistical analysis and modeling.
- Mitarbeiterbindung von Ärztinnen und Ärzten...: Raw data in excel format collected from surveys about employee satisfaction and retention factors.
- HypothesisedModel.png and RetentionModelFit.png: Images representing the hypothesized model of employee retention and its fitness, respectively.
- README.md: This document, explaining the repository contents and instructions.

## **Data Preprocessing**
- Data was cleaned to remove unnecessary columns and encode categorical variables.
- German responses were translated and encoded appropriately using one-hot encoding and label encoding.
- Missing values and outliers were handled, and variables were prepared for analysis.

## **Descriptive Analysis**
- Summary statistics were computed for all the variables, and the structure of the dataset was understood.
- Correlation analysis was conducted to identify relationships between variables.

## **Inferential Statistics**
- Linear regression models were constructed to determine the impact of various factors on different aspects of employee retention: affective, cognitive, normative, and contractual retention.
- A stepwise regression approach was used to refine the models and select the most influential factors.
- MANCOVA was performed to analyze the impact of factors holistically.
- Variance Inflation Factor (VIF) was calculated to check for multicollinearity.

## **Regression Models and MANCOVA**
- Separate regression models were built for affective, cognitive, normative, and contractual retention factors to identify significant predictors.
- MANCOVA was used to analyze the influence of demographic and work-related factors on the retention components collectively.
- The results provided insight into the relative importance of different factors on employee retention post-COVID-19.

## **Conclusions**
- The analysis highlighted key factors contributing to employee retention and identified potential areas for improvement in employment practices.
- The findings can assist HR departments and policymakers in devising strategies to enhance employee retention, especially in times of crisis such as the COVID-19 pandemic.

## **Prerequisites**

Before running the scripts in this repository, ensure that you have the following environments and packages installed:
Python
- numpy
- pandas
- matplotlib
- seaborn
- statsmodels
- sklearn
- semopy for Structural Equation Modeling
- pingouin for statistical analysis

To install Python packages, use:
    
    pip install numpy pandas matplotlib seaborn statsmodels scikit-learn semopy pingouin
