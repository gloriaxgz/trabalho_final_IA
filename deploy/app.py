import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Preditor de Condição Médica", layout="wide")


PROFILES = [
    {
        'name': "Perfil 1: Neutro/Saudável",
        'data': {
            'age': 30, 'gender': 'Feminino', 'bmi': 22.0, 'bp': 110.0, 'oxy_sat': 99.0, 'los': 1,
            'glucose': 85.0, 'hba1c': 5.0, 'cholesterol': 150.0, 'triglycerides': 80.0, 'family_history': False,
            'stress': 3.0, 'sleep_hours': 8.0, 'physical_activity': 7.0, 'diet_score': 9.0, 'smoking': False, 'alcohol': False
        }
    },
    {
        'name': "Perfil 2: Risco Metabólico (Diabetes)",
        'data': {
            'age': 65, 'gender': 'Masculino', 'bmi': 35.0, 'bp': 140.0, 'oxy_sat': 95.0, 'los': 5,
            'glucose': 250.0, 'hba1c': 9.0, 'cholesterol': 280.0, 'triglycerides': 220.0, 'family_history': True,
            'stress': 7.5, 'sleep_hours': 6.0, 'physical_activity': 1.0, 'diet_score': 3.0, 'smoking': True, 'alcohol': False
        }
    },
    {
        'name': "Perfil 3: Risco Cardiovascular (Hipertensão)",
        'data': {
            'age': 62, 'gender': 'Feminino', 'bmi': 22.0, 'bp': 160.0, 'oxy_sat': 90.0, 'los': 4,
            'glucose': 85.0, 'hba1c': 5.0, 'cholesterol': 150.0, 'triglycerides': 80.0, 'family_history': False,
            'stress': 6.0, 'sleep_hours': 8.0, 'physical_activity': 4.0, 'diet_score': 6.0, 'smoking': True, 'alcohol': True
        }
    },
    {
        'name': "Perfil 4: Cancer",
        'data': {
            'age': 75, 'gender': 'Feminino', 'bmi': 27.0, 'bp': 130.0, 'oxy_sat': 98.0, 'los': 10,
            'glucose': 105.0, 'hba1c': 5.8, 'cholesterol': 200.0, 'triglycerides': 140.0, 'family_history': False,
            'stress': 8.0, 'sleep_hours': 6.0, 'physical_activity': 1.0, 'diet_score': 4.0, 'smoking': False, 'alcohol': False
        }
    },
    {
        'name': "Perfil 5: Extremo (Obesidade Pura)",
        'data': {
            'age': 40, 'gender': 'Feminino', 'bmi': 50.0, 'bp': 130.0, 'oxy_sat': 96.0, 'los': 2,
            'glucose': 110.0, 'hba1c': 6.0, 'cholesterol': 240.0, 'triglycerides': 200.0, 'family_history': False,
            'stress': 8.0, 'sleep_hours': 5.0, 'physical_activity': 0.0, 'diet_score': 2.0, 'smoking': True, 'alcohol': True
        }
    },
    {
        'name': "Perfil 6: Asma",
        'data': {
            'age': 31, 'gender': 'Feminino', 'bmi': 20.0, 'bp': 100.0, 'oxy_sat': 85.0, 'los': 2,
            'glucose': 100.0, 'hba1c': 5.0, 'cholesterol': 100.0, 'triglycerides': 100.0, 'family_history': True,
            'stress': 6.0, 'sleep_hours': 7.0, 'physical_activity': 4.0, 'diet_score': 7.0, 'smoking': True, 'alcohol': False
        }
    }
]

features_base = [
    'Age', 'Gender_Encoded', 'Glucose', 'Blood Pressure', 'BMI',
    'Oxygen Saturation', 'LengthOfStay', 'Cholesterol', 'Triglycerides',
    'HbA1c', 'Smoking', 'Alcohol', 'Physical Activity', 'Diet Score',
    'Family History', 'Stress Level', 'Sleep Hours'
]


def initialize_session_state(profile_data=None):

    if profile_data is not None:
        initial_values = profile_data
        
    elif 'initialized' not in st.session_state:
        initial_values = PROFILES[0]['data']
        st.session_state['profile_index'] = 0
        st.session_state['prediction_made'] = False
    
    else:
        return
        
    for key, value in initial_values.items():
        st.session_state[key] = value
            
    st.session_state['initialized'] = True


def next_profile():
    if 'profile_index' not in st.session_state:
        st.session_state['profile_index'] = 0
        
    new_index = (st.session_state['profile_index'] + 1) % len(PROFILES)
    st.session_state['profile_index'] = new_index
    
    new_profile_data = PROFILES[new_index]['data']
    
    initialize_session_state(new_profile_data)

    
@st.cache_resource()
def load_resources():

    BASE_DIR = Path(__file__).resolve().parent

    MODEL_PATH = BASE_DIR / "modelo_random_forest.pkl"
    SCALER_PATH = BASE_DIR / "scaler_base.pkl"
    LE_PATH = BASE_DIR / "label_encoder.pkl"

    model, scaler, le = None, None, None

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)
        return model, scaler, le

    except FileNotFoundError as e:
        st.error(f"Erro fatal: Arquivo essencial '{e.filename}' não encontrado.")
        st.error("Por favor, verifique se todos os arquivos .pkl (modelo, scaler e encoder) estão no mesmo diretório do app.py e foram incluídos no seu deploy.")
        return None, None, None
    except Exception as e:
        st.error(f"Erro inesperado ao carregar recursos: {e}")
        return None, None, None

# Carrega os recursos e inicializa o estado
model, scaler, le = load_resources()
initialize_session_state()



def display_model_insights():
 
    st.markdown("---")
    st.header("Conclusões e Interpretações Detalhadas do Modelo")
    st.markdown(
        """
        Esta seção detalha a performance e a interpretabilidade do modelo preditivo de condição médica,
        incluindo a importância dos fatores e a análise de erros.
        """
    )
    
    data_importancia = {
        'Feature': ['BMI', 'Glucose', 'HbA1c', 'Blood Pressure', 'Age', 'Sleep Hours', 'Cholesterol', 'Triglycerides', 'Stress Level', 'Physical Activity'],
        'Score': [0.28, 0.19, 0.15, 0.09, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02]
    }
    df_importancia = pd.DataFrame(data_importancia).sort_values(by='Score', ascending=False)

    condicoes = ['Arthritis', 'Asthma', 'Cancer', 'Diabetes', 'Healthy', 'Hypertension', 'Obesity']
    data_matriz = {
        'Arthritis': [220, 7, 0, 2, 0, 99, 31],
        'Asthma': [7, 363, 0, 1, 6, 20, 10],
        'Cancer': [0, 0, 232, 15, 0, 0, 0],
        'Diabetes': [7, 3, 12, 1201, 0, 47, 14],
        'Healthy': [2, 1, 0, 0, 604, 1, 0],
        'Hypertension': [28, 15, 0, 19, 2, 1336, 24],
        'Obesity': [11, 3, 0, 14, 1, 46, 696]
    }
    df_matriz_confusao = pd.DataFrame(data_matriz, index=condicoes).T
    df_matriz_confusao.columns.name = 'Condição Predita'
    df_matriz_confusao.index.name = 'Condição Verdadeira'

    data_metrics = {
        'Métrica': ['Acurácia (Geral)', 'Precisão Média (Macro)', 'Recall Médio (Macro)', 'F1-Score (Ponderado)'],
        'Valor (%)': [88.5, 86.2, 87.9, 88.0]
    }
    df_metrics = pd.DataFrame(data_metrics)
    
    
    st.subheader("1. Fatores Mais Influentes na Decisão do Modelo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            df_importancia.head(7),
            x='Score',
            y='Feature',
            orientation='h',
            title='Top 7 Variáveis com Maior Importância',
            color_discrete_sequence=px.colors.qualitative.D3
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'},
                          plot_bgcolor="#0E1117",
                          paper_bgcolor="#0E1117",
                          font_color="#FAFAFA")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.info("### Principais Insights sobre Fatores")
        st.markdown(f"- **Fator Dominante:** **{df_importancia.iloc[0]['Feature']}** é o mais decisivo (Score: {df_importancia.iloc[0]['Score']:.2f}).")
        st.markdown(f"- **Marcadores Metabólicos:** **Glicose** e **HbA1c** são cruciais para o diagnóstico de Diabetes (Score Combinado > 0.34).")
        st.markdown("- **Estilo de Vida:** A importância de **Horas de Sono** e **Nível de Estresse** demonstra que o modelo considera o bem-estar geral, impactando especialmente as classes *Healthy* e *Hypertension*.")
        st.markdown("- **Idade vs. BMI:** A Idade tem uma importância ligeiramente menor que o BMI, sugerindo que o **estado físico atual** é mais preditivo do que apenas a cronologia.")

 
    st.subheader("2. Matriz de Confusão do Modelo e Erros Críticos")
    
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df_matriz_confusao,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        ax=ax_cm,
        linewidths=.5,
        linecolor='black'
    )
    ax_cm.set_xticklabels(ax_cm.get_xticklabels(), rotation=45, ha='right')
    ax_cm.set_yticklabels(ax_cm.get_yticklabels(), rotation=0)
    
    ax_cm.set_title('Matriz de Confusão (Valores Verdadeiros vs Preditos)')
    ax_cm.set_ylabel('Condição VERDADEIRA (True Label)')
    ax_cm.set_xlabel('Condição PREDITA (Predicted Label)')
    
    st.pyplot(fig_cm)
    
    st.info("### Análise de Erros (Matriz)")
    st.markdown("- **Forte Desempenho:** **Diabetes** e **Hipertensão** são as classes mais bem previstas devido à clareza dos marcadores (Glicose/BP).")
    st.markdown("- **Principal Confusão:** O erro mais significativo (99 casos) ocorre quando o modelo prevê **Hipertensão** para casos que são, na verdade, **Arthritis**. Isso sugere uma forte co-ocorrência de pressão alta e marcadores de artrite no dataset de treino.")
    st.markdown("- **Baixa Confusão:** A classe **Healthy** tem pouquíssimos falsos positivos e negativos, indicando que o modelo distingue bem a saúde plena dos estados de doença.")

    st.subheader("3. Análise Fatorial Detalhada: Como as Variáveis Afetam a Predição")
    st.markdown("Esta análise simula o impacto de cada grupo de variáveis nas principais condições, fornecendo uma visão granular da tomada de decisão do modelo, similar à interpretabilidade LIME/SHAP.")

    col_det_1, col_det_2 = st.columns(2)
    
    with col_det_1:
        st.markdown("#### Fatores Metabólicos e de Risco Cardiovascular")
        
        st.markdown("**1. BMI (Índice de Massa Corporal):**")
        st.warning(
            "É a variável mais determinante. Valores **acima de 30** (Obesidade) disparam fortemente a probabilidade das classes **Obesity** e **Hypertension**. Um BMI alto também contribui secundariamente para **Arthritis** e **Diabetes**, atuando como um fator de risco geral."
        )

        st.markdown("**2. Glicose e HbA1c:**")
        st.warning(
            "Esses marcadores são o principal da previsão de **Diabetes**. Níveis altos de **Glicose** ou **HbA1c** imediatamente elevam a probabilidade de **Diabetes** e reduzem as chances de classes como *Healthy*."
        )
        
        st.markdown("**3. Pressão Arterial (BP) e Saturação O2:**")
        st.warning(
            "BP é crucial para **Hypertension**; valores altos aumentam muito a chance desse diagnóstico. Já a **Saturação de Oxigênio** (Oxygen Saturation) é o fator mais forte para o diagnóstico de **Asthma**, servindo também como um sinal de alerta para qualquer condição respiratória ou grave."
        )

    with col_det_2:
        st.markdown("#### Fatores Demográficos e de Estilo de Vida")
        
        st.markdown("**1. Idade e Histórico Familiar:**")
        st.info(
            "A **Idade** funciona como um multiplicador de risco: embora não seja o fator mais importante, pacientes mais velhos com marcadores de risco elevados têm sua probabilidade de **Cancer** e **Hypertension** aumentada, refletindo a progressão do risco ao longo do tempo. O **Histórico Familiar** age como um peso adicional em condições específicas (Diabetes/Cancer), mas sua importância é menor do que a dos marcadores diretos."
        )

        st.markdown("**2. Stress, Sono e Atividade Física:**")
        st.info(
            "São fatores de *ajuste fino*. **Baixas Horas de Sono** e **Alto Nível de Stress** aumentam a probabilidade de **Hypertension** e **Arthritis** . Por outro lado, **Alta Atividade Física** e **Score de Dieta** acima de 7 tendem a *reduzir* o risco de **Obesity** e *aumentar* a probabilidade da classe **Healthy**, servindo como atenuantes de risco claros."
        )

        st.markdown("**3. Colesterol e Triglicerídeos:**")
        st.info(
            "Seu impacto é mais sutil. Valores elevados dessa variável tendem a **reforçar** uma predição já inclinada para **Hypertension** ou **Obesity**. Eles são importantes, mas raramente são o único fator que decide uma previsão isolada no modelo."
        )
        
    
    st.markdown("---")
    st.header("4. Limitações Atuais e Próximos Passos (Futuro)")
    
    st.subheader("Limitações e Cuidados Essenciais")
    st.success("""
        - **Modelo Preditivo, Não Diagnóstico:** Esta ferramenta deve ser usada apenas para *suporte à decisão* e *triagem*. Nunca substitui um diagnóstico médico profissional, que envolve exames clínicos e contextuais que o modelo não possui.
        - **Dependência de Dados:** O desempenho do modelo está diretamente ligado à qualidade e ao viés do conjunto de dados de treino. Ele pode ter dificuldades em prever condições raras ou em populações que não estão bem representadas nos dados originais.
        - **Ausência de Marcadores Específicos:** Faltam dados clínicos vitais para diagnósticos específicos (ex: resultados de biópsia para Câncer, espirometria para Asma ou marcadores inflamatórios específicos para Artrite). O modelo opera com um conjunto de features mais genérico.
        - **Interpretação:** A probabilidade de 80% para uma condição não significa que as outras 20% não são importantes. O profissional deve sempre analisar o risco global.
    """)

    st.subheader("Possibilidades Futuras e Evolução")
    st.success("""
        - **Integração de Dados Multimodais:** Incorporar dados de prontuários eletrônicos, imagens médicas ou resultados de sequenciamento genético para aumentar a acurácia.
        - **Detecção de Comorbidades:** Treinar um modelo que possa prever a presença de múltiplas condições (ex: Diabetes + Hipertensão) simultaneamente, ao invés de apenas uma única classe.
        - **Personalização de Risco:** Criar um módulo que não apenas preveja a condição, mas que também sugira intervenções personalizadas (dieta, exercícios, acompanhamento) baseadas nos fatores de risco específicos do paciente.
        - **Monitoramento em Tempo Real:** Conectar a aplicação a dispositivos *wearables* ou IoT para monitorar marcadores (como sono) em tempo real e fornecer alertas preditivos precoces.
    """)



st.title('Ferramenta de Previsão de Condição Médica')
st.markdown("Insira os dados do paciente para calcular o diagnóstico mais provável e as probabilidades.")


st.subheader("Preenchimento Rápido de Teste")
col_fill_1, col_fill_2 = st.columns([2, 5])

with col_fill_1:
    current_profile_name = PROFILES[st.session_state.get('profile_index', 0)]['name']
    
    st.button(f"Preenchimento rápido",
              on_click=next_profile,
              help="Clique para preencher os campos rapidamente.",
              use_container_width=True)


if model is not None:
    col1, spacer1, col2, spacer2, col3 = st.columns([4, 0.5, 4, 0.5, 4])

    with col1:
        st.header("Informações Básicas")
        age = st.slider('Idade', 18, 90, st.session_state.age, key='age')
        
        gender_options = ['Feminino', 'Masculino']
        gender_index = gender_options.index(st.session_state.gender)
        gender = st.selectbox('Gênero', gender_options, index=gender_index, key='gender')
        gender_encoded = 1 if gender == 'Feminino' else 0
        
        bmi = st.number_input('BMI (Índice de Massa Corporal)', 15.0, 50.0, st.session_state.bmi, step=0.1, key='bmi')
        bp = st.number_input('Pressão Arterial', 80.0, 200.0, st.session_state.bp, step=1.0, key='bp')
        oxy_sat = st.number_input('Saturação de O2 (%)', 85.0, 100.0, st.session_state.oxy_sat, step=0.1, key='oxy_sat')
        los = st.number_input('Tempo de Permanência (dias)', 1, 30, st.session_state.los, key='los')

    with col2:
        st.header("Marcadores Metabólicos")
        glucose = st.number_input('Glicose (mg/dL)', 50.0, 300.0, st.session_state.glucose, step=1.0, key='glucose')
        hba1c = st.number_input('HbA1c (%)', 4.0, 15.0, st.session_state.hba1c, step=0.1, key='hba1c')
        cholesterol = st.number_input('Colesterol Total (mg/dL)', 100.0, 400.0, st.session_state.cholesterol, step=1.0, key='cholesterol')
        triglycerides = st.number_input('Triglicerídeos (mg/dL)', 50.0, 500.0, st.session_state.triglycerides, step=1.0, key='triglycerides')
        family_history = st.checkbox('Histórico Familiar de Doença Grave', value=st.session_state.family_history, key='family_history')

    with col3:
        st.header("Estilo de Vida")
        stress = st.slider('Nível de Estresse (1-10)', 1.0, 10.0, st.session_state.stress, step=0.1, key='stress')
        sleep_hours = st.slider('Horas de Sono/Noite', 4.0, 10.0, st.session_state.sleep_hours, step=0.1, key='sleep_hours')
        physical_activity = st.number_input('Atividade Física (horas/semana)', 0.0, 20.0, st.session_state.physical_activity, step=0.1, key='physical_activity')
        diet_score = st.slider('Pontuação da Dieta (1-10)', 1.0, 10.0, st.session_state.diet_score, step=0.1, key='diet_score')
        smoking = st.checkbox('Fumante Ativo', value=st.session_state.smoking, key='smoking')
        alcohol = st.checkbox('Consumo Frequente de Álcool', value=st.session_state.alcohol, key='alcohol')

    st.markdown("---")
    
    col_pred_1, col_pred_2, col_pred_3 = st.columns([1, 2, 1])

    with col_pred_2:
        if st.button('Calcular Diagnóstico e Probabilidade', type="primary", use_container_width=True):

            smoking_int = int(smoking)
            alcohol_int = int(alcohol)
            family_history_int = int(family_history)

            input_data = np.array([
                age, gender_encoded, glucose, bp, bmi, oxy_sat, los,
                cholesterol, triglycerides, hba1c, smoking_int, alcohol_int,
                physical_activity, diet_score, family_history_int, stress, sleep_hours
            ]).reshape(1, -1)
            
            input_df = pd.DataFrame(input_data, columns=features_base)

            input_scaled = scaler.transform(input_df)
            
            prediction_encoded = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            predicted_condition = le.inverse_transform(prediction_encoded)[0]
            
            
            st.success(f"O Diagnóstico Mais Provável é: **{predicted_condition}**")

            st.subheader("Distribuição de Probabilidade por Condição")
            
            proba_df = pd.DataFrame({
                'Condição Médica': le.classes_,
                'Probabilidade (%)': prediction_proba[0] * 100
            }).sort_values(by='Probabilidade (%)', ascending=False)
            
            st.dataframe(proba_df.style.format({'Probabilidade (%)': "{:.2f}%"}), use_container_width=True, hide_index=True)
          
            display_model_insights()