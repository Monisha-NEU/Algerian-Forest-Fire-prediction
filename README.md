# Algerian Forest Fires Prediction

This project analyzes the **Algerian Forest Fires dataset** to predict and classify forest fire occurrences based on environmental and meteorological factors. The dataset includes data from two regions of Algeria—Bejaia and Sidi Bel-Abbes—collected over four months in 2012. Multiple machine learning models were used to develop a predictive framework, offering actionable insights into fire occurrences and their severities.

---

## Table of Contents
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [Key Results](#key-results)
- [Conclusion](#conclusion)
- [How to Run the Project](#how-to-run-the-project)
- [References](#references)

---

## Dataset
The dataset was sourced from the **UC Irvine Machine Learning Repository** and contains:
- **244 instances**: 122 each from Bejaia and Sidi Bel-Abbes regions.
- **11 attributes**: Environmental and meteorological features.
- **1 output attribute**: Classifies instances as "fire" or "not fire."

---

## Project Workflow
The project followed a 7-step methodology:
1. **Import and Examine the Dataset**: Loaded and examined the dataset using Python's `pandas` library.
2. **Data Pre-processing**: Cleaned data by removing 2 records with missing values, correcting data types, and saving a new CSV file.
3. **Exploratory Data Analysis (EDA)**: 
   - One-hot encoding for fire classification.
   - Visualizations (histograms, heatmaps, pie charts, box plots) to explore data patterns.
4. **Feature Scaling**: Standardized features to improve model performance.
5. **Split Data**: Divided into 75% training and 25% testing sets.
6. **Model Training**:
   - Used **Linear**, **Ridge**, **Lasso**, and **ElasticNet** regression models.
   - Applied **LassoCV** for 5-fold cross-validation to optimize hyperparameters.
7. **Model Evaluation**: Evaluated using **Mean Absolute Error (MAE)** and **R-squared (R²)** metrics.

---

## Models Used
- **Linear Regression**: A baseline model for understanding linear relationships.
- **Ridge Regression**: Reduces overfitting by penalizing large coefficients (L2 regularization).
- **Lasso Regression**: Performs feature selection by penalizing less impactful features (L1 regularization).
- **ElasticNet Regression**: Combines Lasso and Ridge for robust feature selection and regularization.
- **LassoCV**: Cross-validation for hyperparameter tuning, ensuring robust and optimized results.

---

## Key Results
- **Accuracy**: Achieved R² scores between **87% and 98.4%** across models.
- **Feature Selection**: Dropped highly correlated features (**BUI** and **DC**) with >85% correlation.
- **Temporal Insights**: August recorded the highest number of fires, while September had the least.

---

## Conclusion
This project successfully leveraged machine learning models to predict and classify forest fire occurrences, achieving high accuracy and reliability. The use of feature selection and robust preprocessing techniques contributed to exceptional performance, with R² scores ranging from **87% to 98.4%**. Temporal analysis provided actionable insights, emphasizing the critical months for fire risk. This study demonstrates the effectiveness of machine learning in addressing real-world environmental challenges.

---

## How to Run the Project
1. Clone the repository to your local machine.
2. Ensure you have Python installed along with the required libraries (`pandas`, `scikit-learn`, `matplotlib`, etc.).
3. Load the dataset into the project directory.
4. Run the provided Jupyter Notebook or Python script to replicate the analysis and predictions.

---

## References
- [UC Irvine Machine Learning Repository - Algerian Forest Fires Dataset](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++)
