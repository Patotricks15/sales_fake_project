SELECT 
    p.product_name,
    SUM(f.value) AS total_sales,
    SUM(f.quantity) AS total_quantity_sold
FROM fact_sales f
JOIN dim_product p ON f.id_product = p.id_product
GROUP BY p.product_name
ORDER BY total_sales DESC
