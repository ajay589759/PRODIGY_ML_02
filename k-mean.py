import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Read the data
df = pd.read_csv('Mall_Customers.csv')

# Step 2: Display basic information about the data
print(df.head())  # Display the first few rows of the dataframe
print(df.describe())  # Show basic statistics
print(df.info())  # Show data types and non-null counts

# Step 3: Filter data for spending score > 50
mask = df['Spending Score (1-100)'] > 50
df_score = df[mask]
print(df_score.head())  # Display the first few rows of the filtered dataframe
print(df_score.describe())  # Show basic statistics for the filtered data

# Step 4: Visualize histograms for Age, Annual Income, and Spending Score
plt.figure(figsize=(15, 6))
for n, x in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], 1):
    plt.subplot(2, 3, n)
    sns.histplot(df[x], bins=20)
    plt.title(f'DistPlot of {x}')
plt.tight_layout()
plt.show()

# Step 5: Visualize Age distribution for filtered data
df_score['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Spending Score (51-100): Age Distribution')
plt.show()

# Step 6: Visualize Gender distribution for filtered data
plt.figure(figsize=(15, 4))
sns.countplot(y='Gender', data=df_score)
plt.title('Spending Score (51-100): Gender Distribution')
plt.show()

# Step 7: Visualize Gender distribution for all data
plt.figure(figsize=(15, 4))
sns.countplot(y='Gender', data=df)
plt.title('Spending Score (0-100): Gender Distribution')
plt.show()

# Step 8: Pairplot of Age, Annual Income, and Spending Score
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], kind='reg')
plt.tight_layout()
plt.show()

# Step 9: Scatter plots for Age vs. Annual Income by Gender
plt.figure(figsize=(15, 6))
for gender in ['Male', 'Female']:
    plt.scatter(x='Age', y='Annual Income (k$)', data=df[df['Gender'] == gender], s=200, alpha=0.7, label=gender)
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('Age vs Annual Income wrt Gender')
plt.legend()
plt.show()

# Step 10: Scatter plots for Annual Income vs. Spending Score by Gender
plt.figure(figsize=(15, 6))
for gender in ['Male', 'Female']:
    plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)', data=df[df['Gender'] == gender], s=200, alpha=0.7, label=gender)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Annual Income vs Spending Score wrt Gender')
plt.legend()
plt.show()

# Step 11: Violin and Swarm plots
plt.figure(figsize=(15, 6))
for n, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], 1):
    plt.subplot(1, 3, n)
    sns.violinplot(x=col, y='Gender', data=df, palette='vlag', hue='Gender', legend=False)
    sns.swarmplot(x=col, y='Gender', data=df, color='k', alpha=0.5)
    plt.title(f'Violin & Swarm Plot of {col}')
plt.tight_layout()
plt.show()

# Step 12: Prepare data for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
print(f"X Shape: {X.shape}")
print(X.head())

# Step 13: Determine optimal number of clusters
n_clusters = range(2, 13)
inertia_errors = []
silhouette_scores = []

for k in n_clusters:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia_errors.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X, model.labels_))

print("Inertia:", inertia_errors[:3])
print("Silhouette Scores:", silhouette_scores[:3])

# Step 14: Plot Inertia vs Number of Clusters
plt.figure(figsize=(8, 6))
plt.plot(n_clusters, inertia_errors, marker='o', linestyle='-', color='b')
plt.title('K-Means Model: Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Step 15: Plot Silhouette Scores vs Number of Clusters
plt.figure(figsize=(8, 6))
plt.plot(n_clusters, silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('K-Means Model: Silhouette Scores vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Step 16: Apply final K-Means model with 5 clusters
final_model = KMeans(n_clusters=5, random_state=42, n_init=10)
final_model.fit(X)

labels = final_model.labels_
centroids = final_model.cluster_centers_

print(labels[:5])
print(centroids)

# Step 17: Plot clustering results
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=labels, palette='deep')
sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='black', marker='+', s=500)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Annual Income vs Spending Score")
plt.show()

# Step 18: Cluster analysis
xgb = X.groupby(final_model.labels_).mean()
print(xgb)

# Step 19: Plot side-by-side bar chart of cluster centers
plt.figure(figsize=(8, 6))
bar_width = 0.35
index = range(len(xgb))

plt.bar(index, xgb['Annual Income (k$)'], bar_width, label='Annual Income')
plt.bar([i + bar_width for i in index], xgb['Spending Score (1-100)'], bar_width, label='Spending Score')

plt.xlabel('Clusters')
plt.ylabel('Value')
plt.title('Annual Income and Spending Score by Cluster')
plt.xticks([i + bar_width / 2 for i in index], xgb.index)
plt.legend()
plt.show()
