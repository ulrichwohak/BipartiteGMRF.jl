INSTALL read_stat FROM community;
LOAD read_stat;

CREATE TABLE ceos AS 
SELECT 
    frame_id_numeric,
    person_id,
    year,
    manager_category,
    COUNT(*) OVER (PARTITION BY frame_id_numeric, year) AS n_ceo
FROM read_stat('input/manager-db-ceo-panel/ceo-panel.dta')
WHERE year BETWEEN 1990 AND 2022;

COPY ceos TO 'temp/ceo-panel.parquet' (FORMAT PARQUET);
