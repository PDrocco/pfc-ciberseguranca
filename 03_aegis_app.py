"""
Script 03: Dashboard Interativo

Este script cria a interface com Streamlit e serve como a aplicação principal do sistema de detecção de intrusão.

Funcionalidade da POC:
1. Carrega o modelo de ML pré-treinado.
2. Carrega uma amostra de dados de teste do dataset limpo.
3. Permite ao utilizados clicar em um botão para simular uma análise.
4. Exibe as previsões do modelo e os alertas detectados em uma tabela.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path 
import plotly.express as px

def main():
    SCRIPT_DIR = Path(__file__).parent
    MODELS_DIR = SCRIPT_DIR / 'models'
    DATA_DIR = SCRIPT_DIR / 'processed_data'

    MODEL_PATH = MODELS_DIR / 'decision_tree_model_v1.joblib'
    DATA_PATH = DATA_DIR / 'cleaned_dataset.csv'

    print(f"Carregando o modelo de: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Modelo carregado com sucesso.")

    print(f"Carregando amostra de dados de: {DATA_PATH}")
    df_sample = pd.read_csv(DATA_PATH, nrows=50000)
    print("Amostra de dados carregada com sucesso.")
    
    # 2. Construção da Interface
    st.title(" TOWER: Painel de Detecção de Intrusão (POC)")
    st.write("Clique no botão abaixo para simular a análise de uma amostra de tráfego.")    

    if st.button("Analisar Amostra de Tráfego"):
        with st.spinner('Análisando tráfego...'):
            X_sample = df_sample.drop('Label', axis=1)

            y_pred = model.predict(X_sample)

            results_df = df_sample.copy()
            results_df['Previsão do Modelo'] = y_pred

            #Filtrando para mostrar apenas as linhas onde uma ameaça foi detectada
            detections = results_df[results_df['Previsão do Modelo'] != 'BENIGN']
        
        st.success("Análise concluída!")

        # Verificando se alguma detecção foi encontrada
        if not detections.empty:
            st.warning(f"Atenção! Foram detectadas {len(detections)} ameaças na amostra analisada.")

            st.subheader("Distribuição dos Ataques Detectados.")
            attack_counts = detections['Previsão do Modelo'].value_counts()
            fig = px.pie(
                attack_counts,
                values=attack_counts.values,
                names=attack_counts.index,
                title='Tipos de Ameaças Identificadas'
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Ver detalhes das detecções..."):
                st.dataframe(
                    detections[['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Label', 'Previsão do Modelo']])
        else:
            st.info("Nenhuma ameaça foi detectada nesta amostra.")


if __name__ == "__main__":
    main()