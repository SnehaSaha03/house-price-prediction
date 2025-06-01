# 🏠 House Price Prediction using Machine Learning

This project predicts house prices using a Random Forest model. It includes data cleaning, visualizations, feature selection, model training, and performance evaluation.

---

## 📁 Project Structure

house_price_prediction/
├── data/
│ ├── train.csv
│ ├── test.csv
├── main.py
├── submission.csv
├── scatter_plot.png
├── README.md


---

## 🔧 Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📊 Features Used

We used these features from the dataset:

- `GrLivArea` – Living area (in square feet)
- `OverallQual` – Overall quality score
- `TotalBsmtSF` – Basement area
- `GarageCars` – Number of garage cars
- `YearBuilt` – Year built

---

## 🧹 Data Cleaning

Handled missing values using:

```python
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)
test_data.fillna(test_data.mean(numeric_only=True), inplace=True)

Visualization
We created a scatter plot showing the relation between living area and sale price.


🤖 Model
We used a Random Forest Regressor:
model = RandomForestRegressor(n_estimators=100, random_state=0)

📏 Accuracy Metrics
Model accuracy was measured using:

R² Score: Tells how good the prediction is

RMSE: Root Mean Squared Error

Example output:

mathematica
R² Score: 0.97
Root Mean Squared Error (RMSE): 23000

 Output
The predictions are saved to:
submission.csv

🚀 How to Run
Place train.csv and test.csv into the data/ folder.

Install required libraries:
pip install pandas numpy scikit-learn matplotlib seaborn

Run the script:
python main.py


