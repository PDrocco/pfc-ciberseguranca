"""
Script 01: Pré processamento de dados para o projeto TOWER:

Esse script faz os seguintes passos de pré-processamento:
1.  Localizar todos os arquivos CSV do dataset CIC-IDS-2017.
2.  Unir todos os datasets em um unico dataframe.
3.  Padronização de nomes de colunas.
4.  Remoção de linhas com valores NaN ou infinitos.
5.  Salva o dataset limpo como um novo arquivo CSV para carregar mais rapidamente nos próximos passos.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def main(): 
    # 1. Carregando dados
    print("--- Data Preprocessing ---")
    SCRIPT_DIR = Path(__file__).parent
    RAW_DATA_DIR = SCRIPT_DIR / 'datasets' / 'MachineLearningCVE'

    print(f"Procurando por arquivos CSV em: {RAW_DATA_DIR}")

    # Encontrar e juntar todos os arquivos CSV individuais no diretório de datasets.
    all_csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    print(f"Foram encontrados {len(all_csv_files)} arquivos CSV.")
    dataframe_list = [pd.read_csv(file) for file in all_csv_files]
    print("Juntando todos os datasets...")
    df = pd.concat(dataframe_list, ignore_index=True)
    print(f"Um unico dataframe foi criado com sucesso com as seguintes dimensões: {df.shape}")

    # 2. Limpeza Inicial dos Dados 
    print("\nIniciando a limpeza dos dados...")
    # Padronizando os nomes das colunas removendo espaços em branco extras
    df.columns = df.columns.str.strip()
    print("Nomes das colunas foram limpos e padronizados.")

    # Removendo linhas com valores infinitos ou em falta
    nan_count_before = df.isna().sum().sum()
    numeric_cols = df.select_dtypes(include=np.number)
    inf_count_before = np.isinf(numeric_cols).sum().sum()

    print(f"Encontrados {nan_count_before} valores NaN (em falta).")
    print(f"Encontrados {inf_count_before} valores infinitos.")

    #Substituir os valores infinitos por NaN, e apagar todas as linhas com NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print("Linhas com valores problemáticos foram removidas.")
    print(f"Novas dimensões do DataFrame após a limpeza: {df.shape}")

    # 3. Salvando Dataset Limpo
    # Definindo o caminho e o nome do nosso novo ficheiro de saída
    output_dir = SCRIPT_DIR / 'processed_data'
    output_file_path = output_dir / 'cleaned_dataset.csv'

    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSalvando o dataset limpo em: {output_file_path}")

    #Usando o método to_csv() para salvar o DataFrame.
    # index=False é importante para evitar que o pandas salve o índice do DataFrame como uma coluna extra no ficheiro CSV.
    df.to_csv(output_file_path, index=False)
    print("--- PROCESSO DE PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO! ---")

if __name__ == "__main__":
    main()