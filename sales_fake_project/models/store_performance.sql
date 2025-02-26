SELECT 
    s.store_name,
    SUM(f.value) AS total_sales,
    SUM(f.quantity) AS total_products_sold
FROM fact_sales f
JOIN dim_store s ON f.id_store = s.id_store
GROUP BY s.store_name
