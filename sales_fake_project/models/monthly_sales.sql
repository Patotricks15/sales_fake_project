SELECT 
    d.year,
    d.month,
    SUM(f.value) AS total_sales,
    SUM(f.quantity) AS total_products_sold
FROM fact_sales f
JOIN dim_date d ON f.id_date = d.id_date
GROUP BY d.year, d.month
