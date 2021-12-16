import numpy as np 
import pandas as pd


def split(df, class_feature):
    n_classes = df[class_feature].unique().size
    train_list = []
    test_list = []

    for i in range(n_classes):
        df_sliced = df[df['species'] == i]
        n_rows = df_sliced.shape[0]
        train_list.append(df_sliced.iloc[:n_rows // 2, :])
        test_list.append(df_sliced.iloc[n_rows // 2 :, :])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    X_train = train_df.iloc[:, :-1].to_numpy()
    y_train = train_df.iloc[:, -1].to_numpy()

    X_test = test_df.iloc[:, :-1].to_numpy()
    y_test = test_df.iloc[:, -1].to_numpy()

    return X_train, X_test, y_train, y_test