import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path


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

def initialize_session_state(profile_data=None):
    """
    Inicializa o session state com valores padrão (se for a primeira execução) 
    ou com os dados de um novo perfil (se for chamado por on_click).
    """
    
    if profile_data is not None:
        initial_values = profile_data
        
    elif 'initialized' not in st.session_state:
        initial_values = PROFILES[0]['data']
        st.session_state['profile_index'] = 0
        
    else:
        return
        
    for key, value in initial_values.items():
        st.session_state[key] = value
            
    st.session_state['initialized'] = True


def next_profile():
    """Avança para o próximo perfil na lista e atualiza o session_state."""
    if 'profile_index' not in st.session_state:
        st.session_state['profile_index'] = 0
        
    new_index = (st.session_state['profile_index'] + 1) % len(PROFILES)
    st.session_state['profile_index'] = new_index
    
    new_profile_data = PROFILES[new_index]['data']
    
    initialize_session_state(new_profile_data)


    
@st.cache_resource()
def load_resources():
    """
    Carrega o modelo, scaler e label encoder salvos,
    garantindo que o caminho do arquivo seja sempre encontrado
    usando a localização do próprio app.py (pathlib).
    """

    # Obtém o caminho base do diretório onde este script (app.py) está
    # SÓ FUNCIONA EM AMBIENTES ONDE __file__ ESTÁ DISPONÍVEL (quase todos)
    BASE_DIR = Path(__file__).resolve().parent

    # Define os caminhos completos
    MODEL_PATH = BASE_DIR / "modelo_random_forest.pkl"
    SCALER_PATH = BASE_DIR / "scaler_base.pkl"
    LE_PATH = BASE_DIR / "label_encoder.pkl"

    model, scaler, le = None, None, None

    try:
        # Tenta carregar os arquivos usando o caminho completo e seguro
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)

        # st.success("Recursos de Machine Learning carregados com sucesso.") 
        # (Remover st.success para produção, mas pode ajudar no debug)
        return model, scaler, le

    except FileNotFoundError as e:
        # Exibe o erro de forma clara no Streamlit
        st.error(f"Erro fatal: Arquivo essencial '{e.filename}' não encontrado.")
        st.error("Por favor, verifique se todos os arquivos .pkl (modelo, scaler e encoder) estão no mesmo diretório do app.py e foram incluídos no seu deploy.")
        return None, None, None
    except Exception as e:
        # Captura outros erros (ex: arquivo corrompido)
        st.error(f"Erro inesperado ao carregar recursos: {e}")
        return None, None, None

model, scaler, le = load_resources()
initialize_session_state() 

features_base = [
    'Age', 'Gender_Encoded', 'Glucose', 'Blood Pressure', 'BMI', 
    'Oxygen Saturation', 'LengthOfStay', 'Cholesterol', 'Triglycerides', 
    'HbA1c', 'Smoking', 'Alcohol', 'Physical Activity', 'Diet Score', 
    'Family History', 'Stress Level', 'Sleep Hours'
]

st.title('Ferramenta de Previsão de Condição Médica')
st.markdown("Insira os dados do paciente para calcular o diagnóstico mais provável e as probabilidades.")


st.subheader("Preenchimento Rápido de Teste")
col_fill_1, col_fill_2 = st.columns([2, 5])

with col_fill_1:
    current_profile_name = PROFILES[st.session_state['profile_index']]['name']
    
    st.button(f"Preenchimento Rápido", 
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
            
