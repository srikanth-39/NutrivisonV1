import mysql.connector
import pandas as pd

csv_filename='food_nutrition_per_100g.csv'
try:
    df=pd.read_csv(csv_filename)
    print(f"CSV file '{csv_filename}' loaded successfully.")

except Exception as e:
    print(f"Error loading CSV file '{csv_filename}': {e}")
    exit()

connection=mysql.connector.connect(
    host='localhost',
    user='root',
    password='srikanth123'
)

print("Connected to MySQL server successfully.")


cursor=connection.cursor()
cursor.execute("create database if not exists nutrivision_db")
cursor.execute("use nutrivision_db")
cursor.execute(""" create table if not exists foods_list(
ID  int auto_increment primary key,
food_item varchar(255) not null,
calories_kcal float not null,
protien_g float not null,
carbohydrates_g float not null,
total_fat_g float not null,
sugar_g float not null

)""")


insert_query="""insert into foods_list (food_item, calories_kcal, protien_g,
 carbohydrates_g, total_fat_g, sugar_g) values (%s,%s,%s,%s,%s,%s)"""

records=[tuple(row) for row in df.to_numpy()]

cursor.executemany(insert_query,records)

connection.commit()

print(f"{cursor.rowcount} records inserted successfully into foods_list table.")

cursor.execute("select * from foods_list limit 5")
rows=cursor.fetchall()

print("Sample top 5 records from foods_list table:")

for row in rows:
    print(row)

cursor.close()
connection.close()
print("MySQL connection closed.")


