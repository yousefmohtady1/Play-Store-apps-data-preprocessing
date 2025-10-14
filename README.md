# Play Store Apps Data Analysis and Prediction

## Overview

This Jupyter notebook provides a comprehensive analysis of the Google Play Store dataset, focusing on data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling to forecast app installs. The project demonstrates best practices in handling messy real-world data, including duplicate removal, missing value imputation, categorical encoding, and scaling. Multiple regression models are trained and compared to identify the most effective predictor of app popularity (measured by installs).

Key objectives:

- Clean and prepare the dataset for machine learning.
- Perform EDA to uncover insights into app characteristics and performance.
- Build and evaluate models to predict installs based on features like ratings, reviews, size, and genres.
- Visualize results using interactive tools like Plotly Dash.

The analysis reveals that non-linear models, particularly Random Forest, outperform linear approaches, achieving an R² score of ~0.86.

## Dataset

The dataset is sourced from the [Google Play Store Apps dataset on Kaggle](https://www.kaggle.com/lava18/google-play-store-apps). It includes ~10,841 entries with 13 columns covering app metadata such as:

- **App**: App name
- **Category**: Primary category (e.g., ART_AND_DESIGN, BUSINESS)
- **Rating**: Average user rating (1-5)
- **Reviews**: Number of user reviews
- **Size**: App size (in MB or "Varies with device")
- **Installs**: Number of installs (e.g., "1,000,000+")
- **Type**: Free or Paid
- **Price**: Price in USD
- **Content Rating**: Age appropriateness (e.g., Everyone, Teen)
- **Genres**: Sub-genres (e.g., Art & Design;Creativity)
- **Last Updated**: Date of last update
- **Current Ver**: Current version
- **Android Ver**: Minimum Android version required

After preprocessing, the dataset is reduced to ~10,358 unique rows, with engineered features like normalized app size, Android version numbers, and app age in years.

## Key Steps in the Notebook

### 1. Data Understanding

- Load the CSV and inspect structure (`df.head()`, `df.info()`).
- Identify data types, missing values, and anomalies (e.g., one misaligned row in the Category column).

### 2. Data Cleaning

- **Duplicates**: Remove 483 duplicate rows based on all columns.
- **Missing Values**: Drop rows with NaN in critical columns (e.g., Rating); impute others where feasible.
- **Feature Engineering**:
  - Convert `Reviews` and `Installs` to integers (remove commas and '+' symbols).
  - Parse `Size` to numeric MB (handle "Varies with device" as median).
  - Extract numeric `Android Ver` (e.g., "4.0.3 and up" → 4.0).
  - Calculate `App Age` in years from `Last Updated` relative to a reference date.
  - One-hot encode `Content Rating`.
  - Binary encode `Category` and `Genres` using `category_encoders`.
- Handle outliers and ensure data integrity.

### 3. Exploratory Data Analysis (EDA)

- Visualize distributions (e.g., ratings, installs) using Matplotlib and Seaborn.
- Explore correlations (e.g., reviews vs. installs) and category-based insights.
- Identify trends, such as higher installs in popular categories like FAMILY and GAME.

### 4. Model Training and Evaluation

- **Target**: Predict `Installs` (log-transformed for better fit).
- **Features**: Encoded categories, ratings, reviews, size, type, price, app age, etc.
- **Preprocessing**: Standard scaling and train-test split (80/20).
- **Models Compared**:
  | Model | R² Score | Mean Squared Error (MSE) | Notes |
  |------------------------|----------|---------------------------|-------|
  | Linear Regression | 0.54 | 30.41B | Baseline; underfits non-linear patterns. |
  | Polynomial Regression (degree=2) | 0.73 | 18.02B | Captures interactions but risks overfitting. |
  | Random Forest (n=200) | **0.86** | **9.46B** | Best performer; handles non-linearity and feature importance robustly. |

- Evaluation metrics: R² and MSE on the test set.
- Feature importance highlights reviews and ratings as top predictors.

### 5. Visualization and Dashboard

- Interactive plots using Plotly Express (e.g., scatter plots of installs vs. ratings).
- A basic Dash app for exploring app metrics (e.g., category-based install distributions).

## Results and Insights

- **Top Predictors**: Reviews and ratings strongly correlate with installs, indicating user engagement drives popularity.
- **Category Trends**: GAME and FAMILY apps dominate installs, while niche categories like EVENTS lag.
- **Model Recommendation**: Use Random Forest for production predictions due to its superior accuracy and interpretability.
- **Limitations**: Dataset is from 2018; real-time updates could enhance relevance. Installs are binned, introducing granularity loss.

## Requirements

To run this notebook, install the following Python packages:

```bash
pip install pandas numpy matplotlib seaborn dash plotly scikit-learn category-encoders
```
