import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class Modelo():
    def __init__(self):
        self.df = None
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.

        Parâmetros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.

        O dataset é carregado com as seguintes colunas: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm e Species.
        """
        # Carregar dataset, ignorando a primeira linha (cabeçalho)
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, header=None, names=names)

    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.

        Verifica a presença de valores ausentes e faz o tratamento adequado.
        Para este exemplo, removemos valores ausentes e convertimos a coluna "Species" para variável categórica.
        """
        # Verificar valores ausentes
        if self.df.isnull().sum().any():
            print("Dados ausentes encontrados! Realizando tratamento...")
            self.df = self.df.dropna()  # Remover linhas com valores ausentes

        # Convertendo a variável "Species" em valores numéricos para a regressão (apenas para fins de exemplo)
        self.df['Species'] = self.df['Species'].astype('category').cat.codes

    def Treinamento(self):
        """
        Treina o modelo de regressão linear.

        Divide os dados em treino e teste, treina o modelo LinearRegression e armazena as previsões.
        """
        # Seleciona as variáveis independentes (features) e dependentes (target)
        X = self.df[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  # Exemplo de variáveis independentes
        y = self.df['SepalLengthCm']  # Variável dependente (target)

        # Dividir os dados em conjunto de treino e teste (80% treino e 20% teste)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar o modelo de regressão linear
        self.model.fit(self.X_train, self.y_train)

    def Teste(self):
        """
        Avalia o desempenho do modelo treinado nos dados de teste.

        Calcula o erro quadrático médio (MSE) e o R2 (coeficiente de determinação).
        """
        # Realizar previsões
        y_pred = self.model.predict(self.X_test)

        # Calcular e imprimir as métricas de desempenho
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R2 Score: {r2}")

    def Train(self):
        """
        Função principal para o fluxo de treinamento do modelo.

        Este método encapsula as etapas de carregamento de dados, pré-processamento e treinamento do modelo.
        """
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.

        # Tratamento de dados opcional, pode ser comentado se não for necessário
        self.TratamentoDeDados()

        self.Treinamento()  # Executa o treinamento do modelo
        self.Teste()  # Avalia o modelo após o treinamento


# Instanciando e treinando o modelo
modelo = Modelo()
modelo.Train()