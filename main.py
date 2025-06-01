import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error



train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data.fillna(train_data.mean(numeric_only=True), inplace=True)
test_data.fillna(test_data.mean(numeric_only=True), inplace=True)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=train_data['GrLivArea'], y=train_data['SalePrice'])
plt.title('Living Area vs Sale Price')
plt.xlabel('Ground Living Area (sq ft)')
plt.ylabel('Sale Price')
plt.savefig("scatter_plot.png")
plt.show()


features = ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'YearBuilt']
X = train_data[features]
y = train_data['SalePrice']
X_test = test_data[features]


model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)


predictions = model.predict(X_test)
train_predictions = model.predict(X)

print("R² Score:", r2_score(y, train_predictions))
import numpy as np
rmse = np.sqrt(mean_squared_error(y, train_predictions))
print("Root Mean Squared Error (RMSE):", rmse)


output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)
print("✅ Predictions saved to 'submission.csv'!")

print(train_data.head())
