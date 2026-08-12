import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def retrive(name):
    # Connect to Neon using the environment variable
    connection = psycopg2.connect(
        os.getenv("DATABASE_URL")
    )
    
    cursor = connection.cursor()
    
    # Query your table directly (Neon is already connected to your specific DB via the URL)
    cursor.execute("SELECT * FROM foods_list WHERE food_item = %s", (name,))
    data = cursor.fetchall()
    
    cursor.close()
    connection.close()
    return data