import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from MachineLearning.Data.DataRegression import InsuranceDataProvider


df = pd.DataFrame(
    [(d.Age, d.Premium) for d in 
     InsuranceDataProvider.GetLinearInsuranceData()],
    columns=["Age", "Premium"]
)
y = df["Premium"] 
x = df[["Age"]] 
model123 = LinearRegression()
model123.fit(x,y) 
hours = float(input("Enter hours studied: "))
predicted = model123.predict([[hours]])
print(predicted[0])
#df["predicted"] = model123.predict([[hours]])
#print(df["predicted"])