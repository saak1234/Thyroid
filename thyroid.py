import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('data.csv')
df = pd.get_dummies(df)

features = df.drop('hypethyroid',axis=1)
target = df['hypethyroid']

X_train,X_test,y_train,y_test =train_test_split(features,target,test_size=0.2,random_state=42)

bayes_net = BayesianNetwork([('T3','hypethyroid'), ('T4','hypethyroid'), ('TSH','hypethyroid'),('Goiter','hypethyroid')])

bayes_net.fit(pd.concat([X_train, y_train], axis=1), estimator=MaximumLikelihoodEstimator)

inference_engine = VariableElimination(bayes_net)

def predict_hypethyroid(input_data):
    results = inference_engine.map_query(variables=['hypethyroid'],evidence=input_data)
    return results['hypethyroid']

predictions = [predict_hypethyroid(x) for x in X_test.to_dict(orient='records')]

accuracy = accuracy_score(y_test,predictions)
print(f'Model Accuracy: {accuracy:.2f}')