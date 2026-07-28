import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("1) iris.csv")
print(data.describe())
print(data.corr(numeric_only=True))

#Histogram
plt.figure()
plt.hist(data["sepal_length"])
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal length")
plt.ylabel("Frequency")

#Boxplot
plt.figure()
plt.boxplot(data["sepal_length"])
plt.title("Sepal length")
plt.ylabel("Sepal length")

#Scatter plot
plt.figure()
plt.scatter(data["sepal_length"],data["petal_length"])
plt.title("Sepals vs Petals")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()