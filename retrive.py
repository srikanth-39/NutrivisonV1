import mysql.connector
def retrive(name):
    connection=mysql.connector.connect(
        host="localhost",
        user="root",
        password="srikanth123"

    )

    cursor=connection.cursor()

    cursor.execute("use nutrivision_db")
    cursor.execute("select * from foods_list where food_item=%s",(name,))
    data = cursor.fetchall()
    return data

