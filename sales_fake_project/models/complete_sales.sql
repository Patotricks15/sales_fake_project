SELECT 
    f.id_sale,
    d.full_date,
    c.name AS customer_name,
    p.product_name,
    p.category,
    s.store_name,
    f.value,
    f.quantity
FROM fact_sales f
JOIN dim_date d ON f.id_date = d.id_date
JOIN dim_customer c ON f.id_customer = c.id_customer
JOIN dim_product p ON f.id_product = p.id_product
JOIN dim_store s ON f.id_store = s.id_store
