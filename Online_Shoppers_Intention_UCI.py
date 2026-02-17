import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


path = r"C:\Users\muham\Downloads\archive (10)\online_shoppers_intention.csv"

try:
    df = pd.read_csv(path)
    print(f" Feedback: Dataset loaded. Number of rows.: {len(df)}")
except Exception as e:
    print(f" eror: {e}")


df_encoded = pd.get_dummies(df, columns=['Month', 'VisitorType', 'Weekend'], drop_first=True)

y = df_encoded['Revenue']
X = df_encoded.drop('revenue', axis=1, errors='ignore') # Küçük harf ihtimaline karşı önlem


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train) 


y_pred = model.predict(X_test)
print(f" Doğruluk Oranı: {accuracy_score(y_test, y_pred)}")


plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['There is', 'There is not'],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("What are people most interested in?")
plt.show()




















