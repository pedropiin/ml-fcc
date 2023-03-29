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
import tensorflow as tf

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

def plot_loss(history):
    """
    Função elaborada e disponibilizada pela própria
    documentação do TensorFlow para plotar a perda 
    a cada epoch (ciclo de aprendizagem)
    """
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Binary crossentropy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(history):
    """
    Função elaborada e disponibilizada pela própria
    documentação do TensorFlow para plotar a precisão 
    a cada epoch (ciclo de aprendizagem)
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary crossentropy")
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    plt.show()

def train_model(x_train, y_train, num_nodes, dropout_prob, learning_rate, batch_size, epochs):
    """
    Implementação de uma rede neural para classificação
    com o TensorFlow. Temos 3 camadas, sendo a última de output,
    com as 2 primeiras tendo 32 nós, usando a função relu
    como função de ativação. A última camada possui apenas um nó,
    e ativa através de uma função sigmoidal
    """
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation="relu", input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation="relu",),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation="sigmoid",),
    ])

    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=['accuracy']
        )

    history = nn_model.fit(
        x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
        )

    return nn_model, history


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
    # svm_model = SVC()
    # svm_model = svm_model.fit(x_train, y_train)
    # y_pred = svm_model.predict(x_test)


    """
    A precisão do nosso modelo representa quantos dos 
    que rotulamos como positivos são de fato positivos. Já
    o recall nos mostra quanto dos positivos verdadeiros
    foram apontados como positivos pelo programa. Para mais,
    f1-score é uma forma de unir precisão com recall,
    generalizando o report.
    """
    # print(classification_report(y_test, y_pred))


    """
    Realizando grid search para descobrir através de múltiplos
    testes qual a melhor combinação de configuração para o nosso
    modelo.
    """
    least_val_loss = float('inf')
    least_loss_model = None
    epochs=100
    for num_nodes in [16, 32, 64]:
        for dropout_prob in [0, 0.2]:
            for learning_rate in [0.1, 0.005, 0.001]:
                for batch_size in [32, 64, 128]:
                    print(f"Number of nodes: {num_nodes}; Dropout probability: {dropout_prob}; Learning rate: {learning_rate}; Batch size: {batch_size}.")
                    nn_model, history = train_model(x_train, y_train, num_nodes, dropout_prob, learning_rate, batch_size, epochs)
                    plot_history(history)
                    val_loss = nn_model.evaluate(x_valid, y_valid)
                    print(val_loss)
                    if val_loss[0] < least_val_loss:
                        least_val_loss = val_loss[0]
                        least_loss_model = nn_model
    
    y_pred = least_loss_model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

    print(classification_report(y_test, y_pred))

main()