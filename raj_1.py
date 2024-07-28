import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz

# Load the dataset from the CSV file
df_main = pd.read_csv("nba2021.csv")

# Filter out rows where the minutes per game (MP) are less than 20
df_main = df_main[df_main.MP >= 20]

# Select important columns and the target variable ("Pos")
imp_columns = ["AST", "TRB", "ORB", "BLK", "DRB", "3PA"]
classlabels = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
df = df_main[imp_columns + ["Pos"]]

# Round numeric columns to two decimal places for consistency
df = df.round({"TRB": 2, "ORB": 2, "AST": 2, "BLK": 2, "3PA": 2, "DRB": 2})

# Ignore warnings (optional, for presentation purposes)
warnings.filterwarnings("ignore")

# Create a copy of the dataframe for the RandomForestClassifier
df_rfc = df.copy()

# Prepare the features (X) and target labels (y) for the model
x = df_rfc.drop("Pos", axis=1)
y = df_rfc.loc[:, "Pos"]
y = y.map(classlabels).values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25)

# Standardize the features using StandardScaler
standardScaler = StandardScaler()
X_scaled = standardScaler.fit(X_train)
X_train_scaled = X_scaled.transform(X_train)
X_test_scaled = X_scaled.transform(X_test)

# Initialize variables to store the best model and its performance
best_score = 0
best_test_results = []
best_model = None

# Loop to find the best hyperparameter (number of estimators) for the RandomForestClassifier
for _ in range(25, 125):
    rfc = RandomForestClassifier(
        n_estimators=_, max_depth=7, bootstrap=True, criterion="entropy"
    )
    rfc = rfc.fit(X_train_scaled, y_train)
    results = rfc.predict(X_test_scaled)
    current_score = accuracy_score(y_test, results)
    if best_score < current_score:
        best_score = current_score
        best_test_results = results
        best_model = rfc

# Predict the labels for the training set using the best model found
traningResults = best_model.predict(X_train_scaled)

# Print the training and test set scores
print("Training score: {:.3f}".format(accuracy_score(y_train, traningResults)))
print("Test set score: {:.3f}".format(best_score))

# Print the confusion matrix
print("Confusion matrix:")
print(
    pd.crosstab(
        y_test.squeeze(),
        best_test_results,
        rownames=["True"],
        colnames=["Predicted"],
        margins=True,
    )
)

# Perform cross-validation to assess model performance
scores = cross_val_score(best_model, x, y, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
