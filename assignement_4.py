from library.classifiers import *
from library.splitter_df import split

df = sns.load_dataset("iris")

encoder = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
decoder = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

df['species'] = df['species'].map(encoder)

df.head()

X_train, X_test, y_train, y_test = split(df, 'species')

BC = Bayes_Classifier()
BC.fit(X_train, y_train, 'cccc', 'u')
BC_pred = BC.evaluate(X_test)
BC_acc = accuracy(BC_pred, y_test)


NBC = NB_classifier()
NBC.fit(X_train, y_train, 'c', 'o')
NBC_pred = NBC.evaluate(X_test)
NBC_acc = accuracy(NBC_pred, y_test)


GNB = Gaussian_NB()
GNB.fit(X_train, y_train)
GNB_pred = GNB.predict(X_test)
GNB_acc = accuracy(GNB_pred, y_test)

print(f'Bayes classifier accuracy is {BC_acc}')
print(f'Naive Bayes classifier accuracy is {NBC_acc}')
print(f'Gaussian Naive Bayes classifier accuracy is {GNB_acc:.2f}')