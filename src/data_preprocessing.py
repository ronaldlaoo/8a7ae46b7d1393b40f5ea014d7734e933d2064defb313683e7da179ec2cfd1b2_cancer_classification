import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
os.getcwd()
os.chdir("..")

def load_dataset():
    df = pd.read_csv('data/raw/cancer_dataset.csv').drop(columns=['Unnamed: 0'])
    X = df.drop(columns=['target'])
    y = df['target']
    
    column_names = X.columns

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use saved column names here
    X_scaled = pd.DataFrame(X_scaled, columns=column_names)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
