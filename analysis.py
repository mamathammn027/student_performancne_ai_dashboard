import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(file_path):

    print("📊 Running Data Analysis...")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    print("\nSummary:")
    print(df.describe())

    plt.figure(figsize=(12,10))

    # 1. Score Distribution
    plt.subplot(2,2,1)
    sns.histplot(df.iloc[:, -1], kde=True, color='green')
    plt.title("Score Distribution")

    # 2. Correlation
    plt.subplot(2,2,2)
    num = df.select_dtypes(include='number')
    sns.heatmap(num.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")

    # 3. Scatter
    plt.subplot(2,2,3)
    sns.scatterplot(x=num.iloc[:,0], y=num.iloc[:,-1])
    plt.title("Feature vs Score")

    # 4. Category count
    plt.subplot(2,2,4)
    cat = df.select_dtypes(include='object').columns
    if len(cat) > 0:
        sns.countplot(x=df[cat[0]])
        plt.title(f"{cat[0]} Distribution")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_data("dataset/student_data.csv")