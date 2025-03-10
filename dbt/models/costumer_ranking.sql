SELECT 
    c.name AS customer_name,
    SUM(f.value) AS total_spent,
    COUNT(f.id_sale) AS number_of_sales
FROM fact_sales f
JOIN dim_customer c ON f.id_customer = c.id_customer
GROUP BY c.name
ORDER BY total_spent DESC
