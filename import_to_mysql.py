import pandas as pd
import mysql.connector

# Read CSV
df = pd.read_csv("data/customer_purchase_data.csv")

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="RetailAnalytics"
)
cursor = conn.cursor()

# Insert each row
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO Customers 
        (Age, Gender, AnnualIncome, NumberOfPurchases, ProductCategory, 
         TimeSpentOnWebsite, LoyaltyProgram, DiscountsAvailed, PurchaseStatus)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        int(row['Age']),
        str(row['Gender']),
        float(row['AnnualIncome']),
        int(row['NumberOfPurchases']),
        str(row['ProductCategory']),
        float(row['TimeSpentOnWebsite']),
        int(row['LoyaltyProgram']),
        int(row['DiscountsAvailed']),
        int(row['PurchaseStatus'])
    ))

conn.commit()
cursor.close()
conn.close()
print("✅ All rows inserted successfully!")
