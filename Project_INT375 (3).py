# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Style
sns.set(style="whitegrid")

# =========================================
# 2. LOAD DATA (SAFE PATH)
# =========================================
df = pd.read_csv('/Users/rakesh/Desktop/DataSet/Crime_Data_from_2020_to_2024.csv')

print("Initial Shape:", df.shape)
print(df.head())

# =========================================
# 3. DATA CLEANING
# =========================================

# Convert Date
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df = df.dropna(subset=['DATE OCC'])

# Convert Time
df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)
df['Hour'] = df['TIME OCC'].str[:2].astype(int)

# Feature Engineering
df['Year'] = df['DATE OCC'].dt.year
df['Month'] = df['DATE OCC'].dt.month

# Filter only 2020
df = df[df['Year'] == 2020]

print("Cleaned Shape:", df.shape)

# =========================================
# 4. BASIC EDA
# =========================================
print("\nINFO:")
print(df.info())

print("\nSUMMARY:")
print(df.describe())

print("\nMISSING VALUES:")
print(df.isnull().sum())

# =========================================
# 5. VISUALIZATIONS
# =========================================

# ---------- Month Count ----------
if 'Month' in df.columns:
    plt.figure()
    sns.countplot(x='Month', hue='Month', data=df, palette="Set2", legend=False)
    plt.title("Crimes by Month (2020)")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.show()

# ---------- Hour Count ----------
if 'Hour' in df.columns:
    plt.figure()
    sns.countplot(x='Hour', hue='Hour', data=df, palette="coolwarm", legend=False)
    plt.title("Crimes by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    plt.show()

# ---------- Top Crime Types ----------
if 'Crm Cd Desc' in df.columns:
    top_crimes = df['Crm Cd Desc'].value_counts().head(10)
    plt.figure()
    plt.barh(top_crimes.index, top_crimes.values, color='purple')
    plt.title("Top 10 Crime Types")
    plt.xlabel("Count")
    plt.show()

# ---------- Pie Chart ----------
if 'Vict Sex' in df.columns:
    plt.figure()
    df['Vict Sex'].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=['skyblue', 'pink', 'lightgreen']
    )
    plt.title("Victim Gender Distribution")
    plt.ylabel("")
    plt.show()

# ---------- Donut Chart ----------
if 'AREA NAME' in df.columns:
    area_counts = df['AREA NAME'].value_counts().head(5)
    plt.figure()
    plt.pie(area_counts, labels=area_counts.index, autopct='%1.1f%%')
    centre_circle = plt.Circle((0, 0), 0.6, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title("Top Areas (Donut Chart)")
    plt.show()

# ---------- Histogram ----------
if 'Hour' in df.columns:
    plt.figure()
    sns.histplot(df['Hour'], kde=True, bins=24, color='orange')
    plt.title("Crime Hour Distribution")
    plt.show()

# ---------- Scatter ----------
if 'Month' in df.columns and 'Hour' in df.columns:
    plt.figure()
    sns.scatterplot(x='Month', y='Hour', hue='Month', data=df, palette="viridis", legend=False)
    plt.title("Month vs Hour")
    plt.show()

# ---------- Boxplot ----------
if 'Month' in df.columns and 'Hour' in df.columns:
    plt.figure()
    sns.boxplot(x='Month', y='Hour', hue='Month', data=df, palette="Set3", legend=False)
    plt.title("Hour Distribution by Month")
    plt.show()

# ---------- Area Count ----------
if 'AREA NAME' in df.columns:
    plt.figure()
    sns.countplot(
        y='AREA NAME',
        hue='AREA NAME',
        data=df,
        order=df['AREA NAME'].value_counts().head(10).index,
        palette="cool",
        legend=False
    )
    plt.title("Top 10 Crime Areas")
    plt.show()

# ---------- Heatmap ----------
numeric_df = df.select_dtypes(include=np.number)
if not numeric_df.empty:
    plt.figure()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# =========================================
# 6. SKEWNESS
# =========================================
if not numeric_df.empty:
    print("\nSkewness:\n", numeric_df.skew())

# =========================================
# 7. OUTLIERS (IQR)
# =========================================
if not numeric_df.empty:
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) |
                (numeric_df > (Q3 + 1.5 * IQR)))

    print("\nOutliers per column:\n", outliers.sum())
    print("Total rows with outliers:", outliers.any(axis=1).sum())

# =========================================
# 8. MACHINE LEARNING (LINEAR REGRESSION)
# =========================================
if 'Hour' in df.columns:
    model_df = df[['Hour']].copy()
    model_df['Index'] = range(len(model_df))

    X = model_df[['Index']]
    y = model_df[['Hour']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\nSample Predictions:\n", pred[:5])

    # Regression Plot
    plt.figure()
    plt.scatter(X, y, color='blue')
    plt.plot(X, model.predict(X), color='red')
    plt.title("Linear Regression (Crime Trend)")
    plt.xlabel("Index")
    plt.ylabel("Hour")
    plt.show()

# =========================================
# 9. KEY INSIGHTS
# =========================================
print("\nKey Insights:")

if 'Hour' in df.columns:
    print("Peak Crime Hour:", df['Hour'].value_counts().idxmax())

if 'Crm Cd Desc' in df.columns:
    print("Most Common Crime:", df['Crm Cd Desc'].value_counts().idxmax())

if 'AREA NAME' in df.columns:
    print("Most Dangerous Area:", df['AREA NAME'].value_counts().idxmax())
