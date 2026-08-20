import mysql.connector
import pandas as pd

# 🔹 MySQL connection details
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="@Siddhi20", 
    database="loan_approval_db"
)

# 🔹 SQL query
query = "SELECT * FROM loan_applications"

# 🔹 Load SQL data into pandas DataFrame
df = pd.read_sql(query, conn)

conn.close()

print("Data loaded successfully from MySQL")
print(df.head())
print("\nShape:", df.shape)
