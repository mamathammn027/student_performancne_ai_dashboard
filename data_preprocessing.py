import pandas as pd
import numpy as np

def clean_data(input_file, output_file):

    print("📥 Loading dataset...")
    df = pd.read_csv(input_file)

    # Remove spaces in column names
    df.columns = df.columns.str.strip()

    print("Before Cleaning:", df.shape)

    # Handle missing values
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype(str).str.strip().str.title()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove invalid values (marks 0–100)
    target = df.columns[-1]
    df = df[(df[target] >= 0) & (df[target] <= 100)]

    print("After Cleaning:", df.shape)

    # Save cleaned data
    df.to_csv(output_file, index=False)
    print("✅ Cleaned data saved as:", output_file)


if __name__ == "__main__":
    clean_data("dataset/student_data.csv", "dataset/cleaned_data.csv")