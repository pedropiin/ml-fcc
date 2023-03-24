import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def scale_dataset(df, oversample = False):
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y


def main():
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("magic04.data", names = cols)

    df["class"] = (df["class"] == "g").astype(int) # casting 'g' = 1 e 'h' = 0

    # for label in cols[:-1]:
    #     plt.hist(df[df["class"] == 1][label], color="blue", label="gamma", alpha=0.7, density=True)
    #     plt.hist(df[df["class"] == 0][label], color="red", label="hadron", alpha=0.7, density=True)
    #     plt.title(label)
    #     plt.ylabel("Probability")
    #     plt.xlabel(label)
    #     plt.legend()
    #     plt.show()

    train, valid, test = np.split(df.sample(frac = 1), [int(0.6*len(df)), int(0.8*len(df))])

    train, x_train, y_train = scale_dataset(train, oversample = True)
    valid, x_valid, y_valid = scale_dataset(valid, oversample = False)
    test, x_test, y_test = scale_dataset(test, oversample = False)

    # KNN

    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(x_train, y_train)

    y_pred = knn_model.predict(x_test)

    print(classification_report(y_test, y_pred))

main()