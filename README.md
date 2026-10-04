\# Sales \& Demand Forecasting



\## Project Overview



This project focuses on forecasting sales using historical sales data and machine learning techniques.



The goal is to analyze previous sales patterns, identify trends, and build a machine learning model that can predict sales.



\## Objective



\- Analyze historical sales data

\- Identify sales trends

\- Perform data preprocessing

\- Create time-based features

\- Build machine learning forecasting models

\- Evaluate model performance

\- Generate useful business insights



\## Dataset



The project uses historical sales transaction data.



The dataset is stored in:



`dataset/train.csv`



The data contains information such as:



\- Order Date

\- Sales

\- Product information

\- Customer information

\- Category information



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Scikit-learn

\- Random Forest Regression



\## Project Methodology



\### 1. Data Loading



The historical sales dataset is loaded using Pandas.



\### 2. Data Preprocessing



The `Order Date` column is converted into a proper date format and the data is sorted chronologically.



\### 3. Exploratory Data Analysis



Sales data is analyzed to understand historical sales patterns and trends.



A daily sales trend graph is created using Matplotlib.



\### 4. Feature Engineering



Time-based and historical sales features are created, including:



\- Year

\- Month

\- Day

\- Day of Week

\- Previous Month Sales

\- Previous 2 Month Sales

\- Previous 3 Month Sales



\### 5. Machine Learning



Different forecasting approaches were tested.



Random Forest Regression was selected for the final monthly forecasting model.



\### 6. Model Evaluation



The model was evaluated using:



\- Mean Absolute Error (MAE)

\- R² Score



\## Final Model



The final model predicts monthly sales using:



\- Year

\- Month

\- Previous Month Sales

\- Previous 2 Month Sales

\- Previous 3 Month Sales



\### Model Performance



\*\*Mean Absolute Error (MAE):\*\*



`14564.49`



\*\*R² Score:\*\*



`0.412`



The model provides a baseline for monthly sales forecasting and captures meaningful patterns in the historical sales data.



\## Business Insights



\### Highest Sales Month



\*\*November 2018\*\*



Sales:



`117938.16`



\### Lowest Sales Month



\*\*February 2016\*\*



Sales:



`11951.41`



\### Average Monthly Sales



`48613.45`



\## Visualizations



The project contains the following visualizations:



\- Daily Sales Trend

\- 30-Day Sales Forecast

\- Actual vs Predicted Sales

\- Monthly Actual vs Predicted Sales



The charts are stored in the `charts` folder.



\## Project Structure



```text

Sales\_Forecasting\_Project/

│

├── dataset/

│   └── train.csv

│

├── charts/

│   ├── daily\_sales\_trend.png

│   ├── 30\_day\_forecast.png

│   ├── actual\_vs\_predicted.png

│   └── monthly\_actual\_vs\_predicted.png

│

├── sales\_forecasting.py

├── improved\_forecasting.py

├── monthly\_forecasting.py

└── README.md



How to Run



First install the required libraries:

pip install pandas numpy matplotlib scikit-learn



Then run the monthly forecasting model:

python monthly\_forecasting.py



Limitations



The current model is a baseline forecasting model.

Its performance can be improved by using more advanced time-series forecasting techniques, additional business features, and hyperparameter tuning.

Future Improvements

\- Use advanced time-series models

\- Add more historical data

\- Perform hyperparameter tuning

\- Include holidays and promotions

\- Include product-level features

\- Include customer-level features

\- Deploy the forecasting model as a web application



Conclusion



This project demonstrates how historical sales data can be analyzed and used to build a machine learning-based sales forecasting system.

The project includes data preprocessing, exploratory data analysis, feature engineering, machine learning, model evaluation, visualization, and business insights.

