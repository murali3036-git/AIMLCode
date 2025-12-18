import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Age": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "Premium": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
}
df = pd.DataFrame(data)
y = df["Premium"] # label
x = df[["Age"]] # feature

model123 = LinearRegression()
model123.fit(x,y) 
hours = float(input("Enter hours studied: "))
predicted = model123.predict([[hours]])
print(predicted[0])
#df["predicted"] = model123.predict([[hours]])
#print(df["predicted"])