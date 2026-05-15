INSTALL read_stat FROM community;
LOAD read_stat;

CREATE TABLE balance_raw AS
SELECT
    CAST(SUBSTRING(frame_id, 3) AS BIGINT) AS frame_id_numeric,
    originalid,
    foundyear,
    year,
    teaor08_2d,
    teaor08_1d,
    COALESCE(sales, 0) AS sales,
    COALESCE(export, 0) AS export,
    GREATEST(FLOOR(COALESCE(emp, 0)), 1) AS employment,
    COALESCE(tanass, 0) AS tangible_assets,
    COALESCE(ranyag, 0) AS materials,
    COALESCE(wbill, 0) AS wagebill,
    COALESCE(persexp, 0) AS personnel_expenses,
    COALESCE(immat, 0) AS intangible_assets,
    COALESCE(so3_with_mo3, 0) AS state_owned,
    COALESCE(fo3, 0) AS foreign_owned
FROM read_stat('input/merleg-LTS-2023-patch/balance/balance_sheet_80_22.dta')
WHERE year BETWEEN 1990 AND 2022
    AND frame_id != 'only_originalid';

CREATE TABLE sector_mapping AS
SELECT
    frame_id_numeric,
    year,
    CASE
        WHEN teaor08_1d = 'A' THEN 1
        WHEN teaor08_1d = 'B' THEN 2
        WHEN teaor08_1d = 'C' THEN 3
        WHEN teaor08_1d IN ('G', 'H') THEN 4
        WHEN teaor08_1d IN ('J', 'M') THEN 5
        WHEN teaor08_1d = 'K' THEN 9
        WHEN teaor08_1d = 'F' THEN 6
        ELSE 7
    END AS sector
FROM balance_raw;

CREATE TABLE sector_mode AS
SELECT
    frame_id_numeric,
    MODE(sector) AS sector
FROM sector_mapping
GROUP BY frame_id_numeric;

CREATE TABLE balance AS
SELECT
    b.frame_id_numeric,
    b.originalid,
    b.foundyear,
    b.year,
    b.teaor08_2d,
    b.teaor08_1d,
    b.sales,
    b.export,
    b.employment,
    b.tangible_assets,
    b.materials,
    b.wagebill,
    b.personnel_expenses,
    b.intangible_assets,
    b.state_owned,
    b.foreign_owned,
    s.sector,
    b.sales - b.personnel_expenses - b.materials AS value_added,
    CASE WHEN b.sales > 0 THEN LN(b.sales) ELSE NULL END AS lnR,
    CASE WHEN b.sales - b.personnel_expenses - b.materials > 0 THEN LN(b.sales - b.personnel_expenses - b.materials) ELSE NULL END AS lnY,
    CASE WHEN b.employment > 0 THEN LN(b.employment) ELSE NULL END AS lnL
FROM balance_raw b
LEFT JOIN sector_mode s ON b.frame_id_numeric = s.frame_id_numeric;

COPY balance TO 'temp/balance.parquet' (FORMAT PARQUET);
