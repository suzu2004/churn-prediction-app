import pandas as pd
import sqlite3

def load_data():
    conn = sqlite3.connect("database/churn.db")

    query = """
    SELECT 
        tenure,
        MonthlyCharges,
        TotalCharges,
        Contract,
        InternetService,
        Churn
    FROM customer_churn
    WHERE MonthlyCharges > 20
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df