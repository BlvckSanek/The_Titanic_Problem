# The Titanic Problem

These Jupyter Notebooks contain the code and analysis for predicting the survival of passengers aboard the Titanic using machine learning techniques.

## Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, resulting in the deaths of a significant number of passengers and crew.

In this project, we aim to predict which passengers survived the Titanic tragedy based on various attributes such as age, gender, passenger class, and embarkation port. By analyzing these factors, we hope to gain insights into the demographics of the survivors and build a predictive model to determine the likelihood of survival for future passengers.

## Dataset

The dataset used for this analysis is the Titanic dataset, which contains information about passengers onboard the Titanic, including their demographics and whether they survived or not. The dataset is provided in CSV format and is available in the `data` directory.

## Methodology

1. **Data Preprocessing**: We begin by exploring and cleaning the dataset, handling missing values, and encoding categorical variables.

2. **Exploratory Data Analysis (EDA)**: We conduct exploratory analysis to understand the distribution of variables, identify patterns, and visualize relationships between features and survival.

3. **Feature Engineering**: We create new features or transform existing ones to improve the predictive power of our model.

4. **Model Building**: We build several machine learning models using algorithms such as gradient boosting classifier and neural network, to predict passenger survival.

5. **Model Evaluation**: We evaluate the performance of each model using appropriate metrics such as accuracy, precision, recall, and ROC curves.

6. **Hyperparameter Tuning**: We fine-tune the parameters of the best-performing model to optimize its performance further.

## Usage

To run any of the Jupyter Notebooks:

1. Clone the repository to your local machine.
2. Ensure you have Python and Jupyter Notebook installed.
3. Navigate to the directory containing the notebooks.
4. Launch Jupyter Notebook and open example the `The_Titanic_Problem_Part_I.ipynb` file.
5. Follow the instructions within the notebook to execute each cell and reproduce the analysis.

## Dependencies

- Python 3.x
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Results

After analyzing the dataset and building predictive models, we achieved [accuracy of 83.6% on the training] for our base model. Our best-performing model [gradient boosting classifier] which was tuned achieved an accuracy of [71.6% accuracy score] on the test set when submitted to the Kaggle competition.

## Conclusion

In conclusion, this project demonstrates the process of analyzing the Titanic dataset and building machine learning models to predict passenger survival. By leveraging various techniques and algorithms, we can gain valuable insights into historical events and make informed predictions about future outcomes. Well this was my proper attempt on a dataset for my Data Science journey. More to come.

For any questions or feedback, feel free to contact [Sanek/arkohnelsonemmanuel@gmail.com].