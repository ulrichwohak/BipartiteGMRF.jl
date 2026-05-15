using Kezdi
using Dates
using Parquet2

# Load balance sheet data
df_balance = Parquet2.readfile("temp/balance.parquet") |> DataFrame
setdf(df_balance)

# Load CEO panel data
df_ceo = Parquet2.readfile("temp/ceo-panel.parquet") |> DataFrame

# Perform the merge
setdf(leftjoin(df_balance, df_ceo, on=[:frame_id_numeric, :year], makeunique=true))

# Keep only observations with valid CEO assignments
@drop @if ismissing(person_id) 
@drop @if ismissing(frame_id_numeric)
@drop @if ismissing(year)

# Identify first time for each CEO-firm pair
@egen first_time = minimum(year), by(frame_id_numeric, person_id)

# Mark new CEO entries
@generate has_new_ceo = (first_time == year)
@egen has_new_ceo_fy = maximum(has_new_ceo), by(frame_id_numeric, year)

# Calculate CEO spell number (cumulative sum of new CEO indicators)
df_sorted = @with getdf() begin
    @sort frame_id_numeric year
end

# Group by frame_id_numeric and calculate cumulative sum
setdf(df_sorted)
transform!(groupby(getdf(), :frame_id_numeric), 
          :has_new_ceo_fy => cumsum => :ceo_spell)

# Apply filters
@drop @if ceo_spell == 0  # No CEO
# use Ref() to avoid broadcasting over the RHS vector
@drop @if sector ∈ Ref([2, 9])

df = getdf()
Parquet2.writefile("temp/merged-panel.parquet", df)