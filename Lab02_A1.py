import pandas as pd
import numpy as np
df=pd.read_excel("Lab02.xlsx",sheet_name="Purchase_data")
df=df.iloc[:,:5]
X=df.drop(columns=['Payment (Rs)','Customer'])
y=df["Payment (Rs)"]
print(X)
print(y)
rank=np.linalg.matrix_rank(X)
print(rank)
