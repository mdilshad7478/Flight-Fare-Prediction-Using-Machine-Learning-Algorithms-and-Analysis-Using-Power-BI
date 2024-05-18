# Flight-Fare-Prediction-Using-Machine-Learning-Algorithms-and-Analysis-Using-Power-BI

This project focuses on predicting flight fares using diverse machine learning algorithms and presenting insightful analysis through a Power BI dashboard. Leveraging historical flight data, we apply machine learning techniques to accurately forecast flight prices. Additionally, we utilize Power BI to create interactive visualizations and a dashboard. Through this project, users can gain a deeper understanding of flight pricing dynamics, enabling them to make informed decisions when booking flights.

## Data Cleaning and Transformation

The flight dataset underwent comprehensive preprocessing to ensure data quality and suitability for model training:

- **Data Cleaning**: Missing values, duplicates, and irrelevant columns were removed to enhance the integrity of the dataset.
- **Feature Engineering**: New features such as day of the week, month, and time of day were derived from date-time features to capture temporal patterns.
- **Categorical Variable Encoding**: Categorical variables were encoded into numerical format using techniques such as one-hot encoding or label encoding to facilitate model training.
- **Feature Scaling**: Numerical features were scaled to a similar range to prevent bias in model training and to improve convergence speed.

 ## Modelling

- **Linear Regression Model**
- **Decision Tree Regressor**
- **Bagging Regressor**
- **Random Forest Regressor**
- **XGB Regressor**
- **KNeighbors Regressor**
- **ExtraTrees Regressor**
- **Ridge Regression**
- **Lasso Regression**

## Model Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R Squared**

Random Forest Regressor is the best model exhibiting low values for MAE, RMSE, MAPE and high value for R squared.

## Power BI Dashboard and Analysis

In addition to model building, a dynamic Power BI dashboard was developed to conduct in-depth analysis of the flight dataset.

Click [here](https://www.novypro.com/profile_about/mohammed-dilshad?Popup=memberProject&Data=1715967251695x287910008688589730) for live dashboard


Snap of Power BI dashboard:
![Alt Snap of Power BI dashboard:](https://github.com/mdilshad7478/Flight-Fare-Prediction-Using-Machine-Learning-Algorithms-and-Analysis-Using-Power-BI/blob/008ea76dada8f8675fa94667d2b29081eb8fb9db/Power%20BI%20dashboard%20snap.jpg)

The dashboard provides users with a comprehensive array of insights to facilitate informed decision-making when planning travel between major 6 airports in India. The key outputs from the dashboard
include:
1. Cheapest Airline: Identification of the airline offering the most economical fares for the selected route, enabling users to prioritize cost-effectiveness.
2. Fastest Airline: Shows the fastest airline between source and destination that passenger needed,
3. Cheapest Time to Fly: Determination of the optimal time of day or day of the week to book flights for the lowest fares, aiding users in scheduling their
travel efficiently.
4. Best Time to Get Cheapest Ticket: Insights into the ideal booking windows to secure the best deals, allowing users to maximize savings while maintaining
flexibility in their travel plans.
5. Number of Flights by Airline: Understanding of the availability and frequency of flights offered by each airline between the selected source and destination,
facilitating comprehensive assessment of travel options.
6. Price for Each Class: Access to pricing information for different ticket classes, including economy, premium economy, business, and first class, assisting
users in selecting the most suitable option based on preferences and budget.
7. Price Varying Based on Number of Stops: Exploration of how ticket prices vary based on the number of stops, providing insights into the trade-offs
between convenience and cost when selecting flights.
8. Average Duration: Shows the average duration needed to reach destination according to different classes and other factors.


By presenting this information in a user-friendly dashboard format, Power BI enhances user accessibility and understanding, empowering travelers to navigate the
complexities of flight booking with confidence and ease.



