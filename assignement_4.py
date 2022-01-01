from library.classifiers import *
from library.splitter_df import split
from sklearn.model_selection import train_test_split

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

# here I randomly split the iris dataset 

X = df[df.columns[:-1]].to_numpy()
y = df[df.columns[-1]].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

def sim(n_sim):
    results = np.zeros((n_sim, 3))
    
    for i in range(n_sim):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        BC = Bayes_Classifier()
        BC.fit(X_train, y_train, 'cccc', 'u')
        BC_pred = BC.evaluate(X_test)
        results[i, 0] = accuracy(BC_pred, y_test)

        NBC = NB_classifier()
        NBC.fit(X_train, y_train, 'c', 'o')
        NBC_pred = NBC.evaluate(X_test)
        results[i, 1] = accuracy(NBC_pred, y_test)

        GNB = Gaussian_NB()
        GNB.fit(X_train, y_train)
        GNB_pred = GNB.predict(X_test)
        results[i, 2] = accuracy(GNB_pred, y_test)
    return results

res = sim(100)
mean = np.around(res.mean(axis=0), 2)
upper = np.around(res.mean(axis=0) + 1.96 * res.std(axis=0), 2)
lower = np.around(res.mean(axis=0) - 1.96 * res.std(axis=0), 2)
upper[upper > 1] = 1.00 # because the values can be upper than 1

# print the results a table
models = ['Bayes', 'Naive Bayes', 'Gaussian N.B.']
for mo, m, l, u in zip(models, mean, lower, upper):
    print(f'{mo}\t{m}[{l}-{u}]')