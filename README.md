# PRODIGY_ML_02

**Mall Customers Clustering Analysis**
Overview
This project analyzes customer data from a shopping mall to identify distinct customer segments using clustering techniques. The analysis includes data visualization, filtering, and clustering with the K-Means algorithm to gain insights into customer behavior and spending patterns.

Dataset
The dataset used in this analysis is Mall_Customers.csv, which contains information about customers such as:

CustomerID: Unique ID for each customer
Gender: Gender of the customer (Male/Female)
Age: Age of the customer
Annual Income (k$): Annual income of the customer in thousands of dollars
Spending Score (1-100): Spending score assigned to the customer based on their behavior and spending nature
Analysis Steps
Data Reading and Initial Exploration

Load the dataset and display basic information and statistics.
Filter data for customers with a spending score greater than 50.
Data Visualization

Histograms for Age, Annual Income, and Spending Score.
Distribution of Age and Gender for the filtered data.
Pairplot to visualize relationships between Age, Annual Income, and Spending Score.
Scatter plots to analyze the relationship between Age, Annual Income, and Spending Score with respect to Gender.
Violin and Swarm plots to visualize the distribution of Age, Annual Income, and Spending Score by Gender.
Clustering with K-Means

Prepare data for clustering by selecting relevant features.
Determine the optimal number of clusters using inertia and silhouette scores.
Apply the final K-Means model with the chosen number of clusters.
Visualize the clustering results and analyze cluster centers.
Cluster Analysis

Group the data by cluster labels and calculate the mean values for Annual Income and Spending Score.
Plot a side-by-side bar chart to compare cluster centers.
Dependencies
Python 3.x
pandas
matplotlib
seaborn
scikit-learn
