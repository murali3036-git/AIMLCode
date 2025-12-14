import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [25, 30, 35, 40],
    "Salary": [50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)

print("DataFrame:\n", df)

print("\nNames:", df["Name"])
print("Ages:", df.Age)

print("\nFirst row:\n", df.iloc[0])
print("Last two rows:\n", df.tail(2))

print("\nAverage Age:", df["Age"].mean())
print("Maximum Salary:", df["Salary"].max())

print("\nPeople with Salary > 60000:\n", df[df["Salary"] > 60000])

df["Bonus"] = df["Salary"] * 0.1
print("\nDataFrame with Bonus:\n", df)
