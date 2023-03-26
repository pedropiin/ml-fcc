import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.naive_bayes
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def scale_dataset(df, oversample = False):
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    """
    Os valores presentes em cada coluna do dataframe variam
    muito (umas colunas possuem elementos que vão até 200, 
    enquanto outras tem seus dados dentro do intervalo 0-1).
    Então, StandardScaler() centraliza e distribui os dados,
    de modo a faze-los ter um desvio padrão igual a 1.
    """
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    

    """
    Temos mais exemplares de dados do tipo 1 ("gamma") do que
    do tipo 2 ("hadron"). Então RandomOverSampler() analisa e 
    reamostra a classe minoritária de dados através de replicação
    aleatória
    """
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    """
    'X', nesse caso, é uma matriz, pois guarda todos os 
    elementos do dataframe, de todas as colunas, menos a
    última. Já 'Y' é um vetor, já que guarda apenas a última
    coluna do dataframe, referente ao tipo da radiação, isto 
    é 1 ou 0. Assim, np.hstack é chamado pra juntar ambas 
    estruturas após a padronização dos dados em 'X'
    """
    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y

def plot_df(df):
    for label in df.columns[:-1]:
        plt.hist(df[df["class"] == 1][label], color="blue", label="gamma", alpha=0.7, density=True)
        plt.hist(df[df["class"] == 0][label], color="red", label="hadron", alpha=0.7, density=True)
        plt.title(label)
        plt.ylabel("Probability")
        plt.xlabel(label)
        plt.legend()
        plt.show()


def main() -> None:
    cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    df = pd.read_csv("magic04.data", names = cols)

    df["class"] = (df["class"] == "g").astype(int) # casting 'g' = 1 e 'h' = 0

    """
    60% dos dados para treinamento, 20% para validação e os restantes 
    20% para testes
    """
    train, valid, test = np.split(df.sample(frac = 1), [int(0.6*len(df)), int(0.8*len(df))])

    train, x_train, y_train = scale_dataset(train, oversample = True)
    valid, x_valid, y_valid = scale_dataset(valid)
    test, x_test, y_test = scale_dataset(test)


    """
    Implementação de KNN como método de classificação.
    Nesse caso, por testes, viu-se que o intervalo mais 
    preciso é de 7 - 11 vizinhos
    """
    # knn_model = KNeighborsClassifier(n_neighbors=11)
    # knn_model.fit(x_train, y_train)
    # y_pred = knn_model.predict(x_test)

    """
    Implementação de Naive-Bayes como método de classificação.
    Mostrou-se, nesse caso, bem menos efetivo que KNN
    """
    # nb_model = sklearn.naive_bayes.GaussianNB()
    # nb_model = nb_model.fit(x_train, y_train)
    # y_pred = nb_model.predict(x_test)

    """
    Implementação de uma Regressão Logística Multifatorial.
    Tem como base a função sigmoidal. Por mais que permita
    diversos parâmetros, mostrou-se mais efetiva que Naive-Bayes,
    porém menos do que KNN.
    """
    # lg_model = LogisticRegression()
    # lg_model = lg_model.fit(x_train, y_train)
    # y_pred = lg_model.predict(x_test)

    """
    Implementação de uma Máquina de Vetor de Suporte (SVM).
    Tenta achar o hiperplano que melhor divide os pontos 
    para classificá-los. Mostrou-se o método mais efetivo 
    de todos para esse dataset
    """
    svm_model = SVC()
    svm_model = svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_test)

    """
    A precisão do nosso modelo representa quantos dos 
    que rotulamos como positivos são de fato positivos. Já
    o recall nos mostra quanto dos positivos verdadeiros
    foram apontados como positivos pelo programa. Para mais,
    f1-score é uma forma de unir precisão com recall,
    generalizando o report.
    """
    print(classification_report(y_test, y_pred))

main()