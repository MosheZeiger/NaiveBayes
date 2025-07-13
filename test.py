import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30]
})

for index, row in df.iterrows():
    print(f"Row {index}: Name={row['Name']}, Age={row['Age']}")