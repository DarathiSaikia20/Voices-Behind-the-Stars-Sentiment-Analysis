<h1 align="center"> 🛍️ Voices Behind the Stars: Sentiment Analysis of Amazon Product Reviews for Consumer Insight </h1>

<p align="center"> 
<img src="GIF/google play.gif" alt="Animated gif" height="282px">
</p>


## 📚 Table Of Contents

- 📋 **Project Description**
- 💾 **Project Files Description**
- 🧾 **Dataset Contents**
- 📊 **Variable Details Of Dataset**
- ❓ **Problem Statement**
- 🛠 **Technologies Used**
- 🔍 **Steps Involved**
- 📌 **Key Insights**
- 🎯 **Conclusion**

- ## 📋 Project Description
The primary objective of this project is to leverage sentiment analysis, a Natural Language Processing (NLP) technique, to classify Amazon product reviews as Positive, Neutral, or Negative.

In today’s e-commerce environment, consumer reviews are crucial in shaping purchasing behavior. However, manually analyzing vast volumes of customer feedback is not scalable. This project applies supervised machine learning to automate sentiment detection using review text, star ratings, and product metadata.

The solution is designed to provide real-time insights into customer satisfaction, enhance product offerings, and guide marketing strategies with data-driven precision.

********************************************************************************************************************************************************************

##  💾 Project Files Description
This repository includes the following files:

- Jupyter Notebook: Contains code for data cleaning, preprocessing, exploratory data analysis (EDA), feature extraction, model training, and evaluation.
- Dataset File: Amazon product reviews in CSV format with review content and product metadata.
- Visualizations & Outputs: Charts, metrics, and model evaluation results.
- README.md: Detailed documentation of the project.

********************************************************************************************************************************************************************

## 🧾 Dataset Contents
The dataset comprises a single CSV file with 1465 rows and 16 columns, representing Amazon product details and customer reviews. The dataset contains both structured and unstructured data, suitable for supervised machine learning and NLP tasks.

********************************************************************************************************************************************************************

## 📊 Variable Details Of Dataset

| Variable                       | Details                                                                                                         |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------|
| product_id                     | Unique identifier for each product                                                                              |
| product_name                   | Name/title of the product                                                                                       |
| category                       |  Category to which the product belongs                                                                          |
| discounted_price               | Final price after applying discounts                                                                            |
| actual_price                   | Original price before any discount                                                                              |
| discount_percentage            | Percentage of discount offered                                                                                  |
| rating                         | Average user rating for the product                                                                             |
| rating_count                   | Total number of user ratings                                                                                    |
| about_product                  | Description or key details about the product                                                                    |
| user_id                        | Unique identifier for each user/reviewer                                                                        |
| user_name                      | Name of the user who wrote the review                                                                           |
| review_id                      | Unique identifier for each review                                                                               |
| review_title                   | Short summary or title of the review                                                                            |
| review_content                 | Full review text written by the customer                                                                        |
| img_link                       | Link to the product image                                                                                       |
| product_link                   | URL to the product page on Amazon                                                                               |
                                                                                   
********************************************************************************************************************************************************************

## ❓ Problem Statement

In the digital age, millions of customer reviews are generated daily, especially on platforms like Amazon. Understanding the emotional tone behind these reviews is critical for sellers, brands, and marketing teams.

This project aims to automatically classify product reviews into sentiment categories (Positive, Neutral, Negative) using NLP and machine learning. The ultimate goal is to extract meaningful insights from user feedback to support better product development, customer engagement, and service strategies.

********************************************************************************************************************************************************************

## 🛠 Technologies Used

- Programming Language: Python

- Libraries Used:
   - pandas, numpy – Data handling
   - matplotlib, seaborn – Visualization
   - scikit-learn – Model building and evaluation
   - nltk, re, string – Natural Language Processing

- Machine Learning Models:
    - Logistic Regression (with class_weight adjustment)
    - Random Forest (with hyperparameter tuning using GridSearchCV)

- Evaluation Metrics:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - Cross-validation

********************************************************************************************************************************************************************

## 🔍 Steps Involved
1. 📥 Data Collection

    - Source: Kaggle
    - Format: CSV

2. 🧹 Data Preprocessing

    - Handled missing values and text inconsistencies
    - Converted data types
    - Refined features for model compatibility

3. 🔍 Exploratory Data Analysis (EDA)

    - Analyzed sentiment distribution
    - Explored correlation between ratings and sentiments
    - Word clouds and bar plots for text insights

4. 🤖 Model Building & Evaluation

    - Implemented Logistic Regression and Random Forest
    - Handled class imbalance using class_weight='balanced' in Logistic Regression
    - Applied hyperparameter tuning using GridSearchCV for Random Forest
    - Evaluated models using accuracy, F1-score, confusion matrix

********************************************************************************************************************************************************************

## 📌 Key Insights

  - Random Forest outperformed Logistic Regression in terms of accuracy and handling class imbalance.
  - Most reviews were skewed toward the Positive class, requiring careful handling during training.
  - Review text and rating features provided strong predictive signals for sentiment classification.
  - Businesses can gain meaningful insights into customer satisfaction, product feedback, and brand reputation from the model’s outputs.
  - The model is suitable for dashboard integration or real-time review monitoring systems.

********************************************************************************************************************************************************************

## 🎯 Conclusion

The sentiment analysis of Amazon product reviews successfully classified feedback into Positive, Neutral, and Negative categories.
  
  - Random Forest proved to be the most effective model for handling imbalanced data and delivering reliable predictions.
  - The project highlighted how NLP and machine learning can uncover hidden patterns in unstructured text.
  - These insights can significantly help businesses optimize offerings, enhance customer experience, and inform marketing strategies.

By automating sentiment detection, the project offers a scalable solution for analyzing customer feedback—paving the way for smarter decision-making and improved customer relationships.














