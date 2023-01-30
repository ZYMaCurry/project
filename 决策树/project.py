import pandas as pd
data = pd.read_csv('outcome.csv',index_col=0)
age_dict = {'<=30': 0., '[31,40]':1., '>40': 2.}
data['Age'] = data['Age'].map(age_dict)

incoming_dict = {'low': 0., 'medium': 1., 'high': 2.}
data['Incoming'] = data['Incoming'].map(incoming_dict)

student_dict = {'yes': 1., 'no': 0.}
data['Student'] = data['Student'].map(student_dict)

credict_rating_dict = {'fair': 0., 'excellent': 1.}
data['Credit Rating'] = data['Credit Rating'].map(credict_rating_dict)

feature = data[['Age','Incoming','Student','Credit Rating']]
target = data[['Buying']]

from sklearn import tree
model = tree.DecisionTreeClassifier()
model = model.fit(feature,target)
import graphviz
dot_data = tree.export_graphviz(model,out_file=None)
graph = graphviz.Source(dot_data)
graph.render('iris_decision_tree')