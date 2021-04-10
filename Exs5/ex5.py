import pandas as pd

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'


def load_data():
    # Load data
    data = pd.read_csv(DATA_URL)
    
    # Show data info in console
    data.info()

    # Preprocess data and split training/test set
    data = data.drop(columns=['name'])
    X = data.drop(columns=['status'])
    y = data.status

    return train_test_split(X, y, test_size=0.2)


def report_metrics(y_pred, y_test):
    # Report the scores for prediction
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    accurary = accuracy_score(y_test, y_pred)
    
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1-score: {f1}")
    print(f"Accurary: {accurary}")


def classifiy():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Create DecisionTreeClassifier object
    rclf = RandomForestClassifier(n_estimators=1000, random_state=1)

    # Create AdaBoostClassifier object
    abc = AdaBoostClassifier(base_estimator=rclf, 
                             n_estimators=rclf.n_estimators)

    # Train Adaboost Classifer
    aclf = abc.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = aclf.predict(X_test)

    report_metrics(y_pred, y_test)


# Run the program with 'python3 ex5.py' 
classifiy()
