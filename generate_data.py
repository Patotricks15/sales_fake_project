import duckdb
import pandas as pd
import random
import datetime
from faker import Faker

fake = Faker('en_US')

con = duckdb.connect(database='project.db')

# Create dimension and fact tables
con.execute("""
CREATE TABLE dim_customer (
    id_customer INT PRIMARY KEY,
    name VARCHAR,
    gender VARCHAR,
    birth_date DATE,
    city VARCHAR,
    state VARCHAR,
    country VARCHAR
);
""")

con.execute("""
CREATE TABLE dim_product (
    id_product INT PRIMARY KEY,
    product_name VARCHAR,
    category VARCHAR,
    brand VARCHAR,
    price DECIMAL(10,2)
);
""")

con.execute("""
CREATE TABLE dim_date (
    id_date INT PRIMARY KEY,
    full_date DATE,
    day INT,
    month INT,
    year INT,
    quarter INT,
    day_of_week VARCHAR
);
""")

con.execute("""
CREATE TABLE dim_store (
    id_store INT PRIMARY KEY,
    store_name VARCHAR,
    address VARCHAR,
    city VARCHAR,
    state VARCHAR,
    country VARCHAR
);
""")

con.execute("""
CREATE TABLE fact_sales (
    id_sale INT PRIMARY KEY,
    id_customer INT,
    id_product INT,
    id_store INT,
    id_date INT,
    value DECIMAL(10,2),
    quantity INT,
    FOREIGN KEY (id_customer) REFERENCES dim_customer(id_customer),
    FOREIGN KEY (id_product) REFERENCES dim_product(id_product),
    FOREIGN KEY (id_store) REFERENCES dim_store(id_store),
    FOREIGN KEY (id_date) REFERENCES dim_date(id_date)
);
""")


# Generate fake data for the dim_customer table
num_customers = 100
customers = []
for i in range(1, num_customers + 1):
    name = fake.name()
    gender = random.choice(['Male', 'Female'])
    birth_date = fake.date_of_birth(minimum_age=18, maximum_age=80)
    city = fake.city()
    state = fake.state()
    country = "USA"
    customers.append((i, name, gender, birth_date, city, state, country))

customers_df = pd.DataFrame(customers, columns=[
    'id_customer', 'name', 'gender', 'birth_date', 'city', 'state', 'country'
])
con.register('customers_df', customers_df)
con.execute("INSERT INTO dim_customer SELECT * FROM customers_df")

# Generate fake data for the dim_product table
num_products = 20
products = []
categories = ['Electronics', 'Clothing', 'Food', 'Books']
brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD']
for i in range(1, num_products + 1):
    product_name = fake.word().capitalize()
    category = random.choice(categories)
    brand = random.choice(brands)
    price = round(random.uniform(10.0, 1000.0), 2)
    products.append((i, product_name, category, brand, price))

products_df = pd.DataFrame(products, columns=[
    'id_product', 'product_name', 'category', 'brand', 'price'
])
con.register('products_df', products_df)
con.execute("INSERT INTO dim_product SELECT * FROM products_df")

# Generate fake data for the dim_date table (for the year 2022)
start_date = datetime.date(2022, 1, 1)
end_date = datetime.date(2022, 12, 31)
delta = datetime.timedelta(days=1)
dates = []
id_counter = 1
current_date = start_date
while current_date <= end_date:
    day = current_date.day
    month = current_date.month
    year = current_date.year
    quarter = (month - 1) // 3 + 1
    day_of_week = current_date.strftime('%A')
    dates.append((id_counter, current_date, day, month, year, quarter, day_of_week))
    id_counter += 1
    current_date += delta

dates_df = pd.DataFrame(dates, columns=[
    'id_date', 'full_date', 'day', 'month', 'year', 'quarter', 'day_of_week'
])
con.register('dates_df', dates_df)
con.execute("INSERT INTO dim_date SELECT * FROM dates_df")

# Generate fake data for the dim_store table
num_stores = 10
stores = []
for i in range(1, num_stores + 1):
    store_name = "Store " + fake.word().capitalize()
    address = fake.address().replace('\n', ' ')
    city = fake.city()
    state = fake.state()
    country = "USA"
    stores.append((i, store_name, address, city, state, country))

stores_df = pd.DataFrame(stores, columns=[
    'id_store', 'store_name', 'address', 'city', 'state', 'country'
])
con.register('stores_df', stores_df)
con.execute("INSERT INTO dim_store SELECT * FROM stores_df")

# Generate fake data for the fact_sales table
num_sales = 500
sales = []
for i in range(1, num_sales + 1):
    id_customer = random.randint(1, num_customers)
    id_product = random.randint(1, num_products)
    id_store = random.randint(1, num_stores)
    # Select a random date within 2022
    id_date = random.randint(1, len(dates))
    quantity = random.randint(1, 10)
    # Retrieve the product price to calculate the total sale value with a random variation
    price = products_df.loc[products_df['id_product'] == id_product, 'price'].values[0]
    value = round(price * quantity * random.uniform(0.9, 1.1), 2)
    sales.append((i, id_customer, id_product, id_store, id_date, value, quantity))

sales_df = pd.DataFrame(sales, columns=[
    'id_sale', 'id_customer', 'id_product', 'id_store', 'id_date', 'value', 'quantity'
])
con.register('sales_df', sales_df)
con.execute("INSERT INTO fact_sales SELECT * FROM sales_df")

# Test query to verify the inserted data
result = con.execute("SELECT * FROM fact_sales LIMIT 5").fetchall()
print("Some sales records:")
for row in result:
    print(row)