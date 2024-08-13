import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions

# Sidebar for selecting parameters
st.sidebar.title("KNN Parameters")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Classification", "Circles", "Moons"))
n_neighbors = st.sidebar.slider("Number of Neighbors (k)", min_value=1, max_value=10, value=3)
weight = st.sidebar.selectbox("weight function",['uniform','distance'] )
algorithm = st.sidebar.selectbox("choose algorith", ['auto','ball_tree','kd_tree','brute'])

# Generate dataset
if dataset_name == "Classification":
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, class_sep=2)
elif dataset_name == "Circles":
    X, y = make_circles(noise=0.1)
else:  # Moons
    X, y = make_moons(noise=0.2, n_samples=1000)

# Scatter plot of the dataset
st.write("### Dataset Visualization")
fig, ax = plt.subplots()
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax)
# st.pyplot(fig)

# KNN and decision boundary plot
st.write(f"### KNN Decision Boundary with k={n_neighbors}")
knn = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weight, algorithm=algorithm, n_jobs=-1)
knn.fit(X, y)
fig, ax = plt.subplots()
plot_decision_regions(X, y, clf=knn, ax=ax)
st.pyplot(fig)
