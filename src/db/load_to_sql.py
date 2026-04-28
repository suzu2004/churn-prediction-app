import pandas as pd
import sqlite3

df= pd.read_csv("data/raw/Telco-Customer-Churn.csv")
conn = sqlite3.connect("churn.db")

df.to_sql("customer_churn", conn, if_exists="replace", index=False)
print("Loaded into SQL")
