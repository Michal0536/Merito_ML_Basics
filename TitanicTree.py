import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import tree
import numpy as np
import pydotplus
from IPython.display import Image
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv', sep=';')

df['Age'] = df['Age'].str.replace(',', '.').astype(float)

X = df.drop('Survived', axis=1)  
y = df['Survived'] 

X = pd.get_dummies(X, columns=['Pclass', 'Sex', 'Embarked'])

clf = DecisionTreeClassifier(max_depth=5) #MAX DEPTH dla czytelnosci wykresu
model = clf.fit(X, y)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=['Died', 'Survived'],
                                filled=True, rounded=True, special_characters=True)

# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())

plt.figure(figsize=(30,15))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Died', 'Survived'], rounded=True, fontsize=10)
plt.show()

with open("titanicTreeOutput.txt","w") as f:
    f.write(dot_data)