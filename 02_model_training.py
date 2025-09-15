"""
Script 02: Treinamento do Modelo para o projeto TOWER

Esse script faz os seguintes passos:
1. Carrega o dataset pré-processado e limpo.
2. Separa os dados em características (X)  e na variável alvo (y)
3. Divide os dados em conjuntos de treino e de teste
4. Treina um modelo classificador de Árvore de Decisão
5. Avalia a performance do modelo no conjunto de teste.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import os
import joblib
#accuracy_score = porcentagem de qnts acertou
#confusion_matrix = quantas vezes ele confundiu um ataque com tráfego BENIGN
#classification_report = um relatório completo com notas detalhadas para cada tipo de ataque.
# Separa o dataset para treino e outro para test.
def main():
    # 1. Carregando o Dataset Limpo
    SCRIPT_DIR = Path(__file__).parent
    PROCESSED_DATA_DIR = SCRIPT_DIR / 'processed_data'
    CLEANED_DATA_FILE = PROCESSED_DATA_DIR / 'cleaned_dataset.csv'
    print(f"Carregando o dataset limpo de {CLEANED_DATA_FILE}")

    df = pd.read_csv(CLEANED_DATA_FILE)
    print("Dataset carregado com sucesso.")
    print(f"Dimensões dos dados carregados: {df.shape}")

    # 2. Separação de Features (X) e target (y)
    print("\nSeparando os dados em features e target...")

    # X recebe todas as colunas, exceto a coluna 'Label'
    #.drop('Label', axis=1) cria uma cópia do df sem essa coluna.
    #axis=1 especifica que removi uma coluna.
    X = df.drop('Label',axis=1)
    y = df['Label']

    print(f"Dimensões de X: {X.shape}")
    print(f"Dimensões de y: {y.shape}")

    # 3. Divisão dos dados em conjuntos de treino e teste
    print("\nDividindo os dados em conjuntos de treino e teste")
    # random_state = 42 garante que a divisão seja sempre a mesma.
    # test_size=03 = 30% dos dados para o conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Dados divididos com sucesso.")
    print(f"Dimensões de X_train (pistas de treino): {X_train.shape}")
    print(f"Dimensões de X_test (pistas de teste): {X_test.shape}")
    print(f"Dimensões de y_train (respostas de treino): {y_train.shape}")
    print(f"Dimensões de y_test (respostas de teste): {y_test.shape}")

    # 4. Treino do Modelo de Árvore de Decisão
    print("\nIniciando o treino do modelo de Árvore de Decisão...")
    print("Essa etapa pode demorar alguns minutos...")
    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")

    # 5. Avaliação do Modelo no conjunto de teste
    print("\nIniciando a avaliação do modelo no conjunto de teste...")
    y_pred = model.predict(X_test)

    print("Previsões geradas. Calculando as métricas de performance...")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Perfomance do Modelo: {accuracy * 100:.2f}%")

    print("\n--- Relatório de Classificação Detalhado ---")
    print(classification_report(y_test, y_pred))

    # 6. Salvando o modelo treinado
    models_dir = SCRIPT_DIR / 'models'
    model_file_path = models_dir / 'decision_tree_model_v1.joblib'

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, model_file_path)

    print(f"Modelo salvo com sucesso em: {model_file_path}")
    

if __name__ == "__main__":
    main()        