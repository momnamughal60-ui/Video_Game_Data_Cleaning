
import pandas as pd
import numpy as np
import os

# ==========================================
# VIDEO GAME SALES - COMPLETE DATA CLEANING
# ==========================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Create output folder
os.makedirs("Output", exist_ok=True)

# Load original dataset
df = pd.read_csv("Dataset/vgsales.csv")

print("=" * 70)
print("       VIDEO GAME SALES - DATA CLEANING PROJECT")
print("=" * 70)

# ==========================================
# 1. ORIGINAL DATASET INSPECTION
# ==========================================

print("\n1. ORIGINAL DATASET INFORMATION")

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFirst 10 Records:")
print(df.head(10).to_string(index=False))

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nTotal Missing Values:",
      df.isnull().sum().sum())

print("\nDuplicate Records Before Cleaning:",
      df.duplicated().sum())

original_rows = len(df)

# ==========================================
# 2. HANDLING MISSING VALUES
# ==========================================

print("\n" + "=" * 70)
print("2. HANDLING MISSING VALUES")
print("=" * 70)

# Fill missing Year with median
year_median = df["Year"].median()

df["Year"] = df["Year"].fillna(year_median).astype(int)

# Fill missing Publisher with Unknown
df["Publisher"] = df["Publisher"].fillna("Unknown")

print("\nMissing values handled successfully!")

# ==========================================
# 3. REMOVING DUPLICATES
# ==========================================

print("\n" + "=" * 70)
print("3. REMOVING DUPLICATE RECORDS")
print("=" * 70)

duplicates_before = df.duplicated().sum()

df = df.drop_duplicates()

duplicates_removed = duplicates_before - df.duplicated().sum()

print("Duplicates Before:", duplicates_before)
print("Duplicates Removed:", duplicates_removed)
print("Duplicates After:", df.duplicated().sum())

# ==========================================
# 4. DATA STANDARDIZATION
# ==========================================

print("\n" + "=" * 70)
print("4. DATA STANDARDIZATION")
print("=" * 70)

text_columns = ["Name", "Platform", "Genre", "Publisher"]

for column in text_columns:
    df[column] = df[column].str.strip()

print("Extra spaces removed from text columns.")
print("Text standardization completed!")

# ==========================================
# 5. OUTLIER DETECTION
# ==========================================

print("\n" + "=" * 70)
print("5. OUTLIER DETECTION")
print("=" * 70)

sales_columns = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales"
]

for column in sales_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = (
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ).sum()

    print(column, "Outliers Detected:", outliers)

print("\nOutliers were inspected and retained.")
print("Reason: High video game sales can be legitimate values.")

# ==========================================
# 6. DATA NORMALIZATION
# ==========================================

print("\n" + "=" * 70)
print("6. DATA NORMALIZATION")
print("=" * 70)

# Min-Max normalization of Global Sales
minimum = df["Global_Sales"].min()
maximum = df["Global_Sales"].max()

df["Global_Sales_Normalized"] = (
    (df["Global_Sales"] - minimum) /
    (maximum - minimum)
)

print("Global Sales normalized successfully!")
print(df[["Global_Sales", "Global_Sales_Normalized"]].head())

# ==========================================
# 7. DATA REDUCTION
# ==========================================

print("\n" + "=" * 70)
print("7. DATA REDUCTION")
print("=" * 70)

# Create a smaller dataset with useful columns
reduced_df = df[
    ["Name", "Platform", "Year", "Genre",
     "Publisher", "Global_Sales"]
]

print("Original Dataset Columns:", len(df.columns))
print("Reduced Dataset Columns:", len(reduced_df.columns))

reduced_df.to_csv(
    "Output/reduced_dataset.csv",
    index=False
)

print("Reduced dataset saved successfully!")

# ==========================================
# 8. DATA AGGREGATION
# ==========================================

print("\n" + "=" * 70)
print("8. DATA AGGREGATION")
print("=" * 70)

genre_summary = df.groupby("Genre").agg(
    Total_Games=("Name", "count"),
    Total_Global_Sales=("Global_Sales", "sum"),
    Average_Global_Sales=("Global_Sales", "mean")
).reset_index()

print("\nGenre-wise Sales Summary:")
print(genre_summary.to_string(index=False))

genre_summary.to_csv(
    "Output/genre_sales_summary.csv",
    index=False
)

print("\nGenre summary saved successfully!")

# ==========================================
# 9. DATA SPLITTING
# ==========================================

print("\n" + "=" * 70)
print("9. DATA SPLITTING")
print("=" * 70)

# Split dataset into 80% training and 20% testing
shuffled_df = df.sample(frac=1, random_state=42)

split_index = int(len(shuffled_df) * 0.8)

train_df = shuffled_df.iloc[:split_index]
test_df = shuffled_df.iloc[split_index:]

train_df.to_csv("Output/train_data.csv", index=False)
test_df.to_csv("Output/test_data.csv", index=False)

print("Training Records:", len(train_df))
print("Testing Records:", len(test_df))
print("Data splitting completed!")

# ==========================================
# 10. FINAL CLEANED DATASET
# ==========================================

print("\n" + "=" * 70)
print("10. FINAL CLEANED DATASET")
print("=" * 70)

df.to_csv(
    "Output/cleaned_vgsales.csv",
    index=False
)

print("\nFirst 10 Cleaned Records:")
print(df.head(10).to_string(index=False))

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nTotal Missing Values After Cleaning:",
      df.isnull().sum().sum())

print("\nDuplicate Records After Cleaning:",
      df.duplicated().sum())

print("\nOriginal Rows:", original_rows)
print("Final Rows:", len(df))
print("Final Columns:", len(df.columns))

print("\n" + "=" * 70)
print("       DATA CLEANING PROJECT COMPLETED!")
print("=" * 70)

print("\nAll output files saved inside Output folder.")
