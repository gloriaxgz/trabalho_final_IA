# Sistema Inteligente de Previsão de Risco à Saúde

Este projeto consiste no desenvolvimento e deploy de um sistema de Machine Learning (ML) capaz de prever o risco de um indivíduo desenvolver diversas condições médicas (como Diabetes, Hipertensão, Obesidade, etc.) com base em fatores de risco clínicos e comportamentais.

A aplicação final do sistema foi realizada utilizando **Streamlit**, focando em uma experiência interativa e **interpretável** para profissionais de saúde.

---

## Problema e Objetivo

O **problema central** abordado é a avaliação de risco e diagnóstico de saúde, dada a multiplicidade de diagnósticos possíveis em um indivíduo. O objetivo primário é desenvolver um sistema que auxilie na identificação de perfis de alto risco e que justifique a predição através da interpretabilidade.

### Metodologia Inicial (Abordagem Híbrida)

Inicialmente, o projeto foi concebido com uma abordagem de ML de duas etapas para maximizar a interpretabilidade:

1.  **Clusterização (K-Means):** Agrupar indivíduos em perfis de risco latentes (e.g., Alto Risco, Baixo Risco), gerando a feature `Risk_Profile_Cluster`.
2.  **Classificação (Random Forest):** Usar o cluster como uma feature adicional para prever o diagnóstico final.

### Decisão do Modelo Final

Após o treinamento e avaliação, o modelo final escolhido foi o **Random Forest** treinado diretamente com as features originais (Modelo Base).

| Modelo | Acurácia |
| :--- | :--- |
| Abordagem Híbrida (com Cluster) | **90.70\%** |
| Modelo Base (sem Cluster) | **91.22\%** |

Apesar da abordagem híbrida ter sido desenvolvida para aumentar a interpretabilidade, o **Modelo Base (91.22\% de acurácia)** demonstrou um desempenho um pouco superior, sendo o modelo selecionado para o deploy.

---

## Estrutura do Repositório

| Arquivo/Pasta | Descrição |
| :--- | :--- |
| `TrabalhoFinal.ipynb` | **Notebook principal do projeto.** Contém toda a Análise Exploratória de Dados (EDA), pré-processamento, a experimentação com a Clusterização (K-Means) e o treinamento e avaliação do modelo Random Forest final. |
| `trabalho_final.pdf` | PDF do desenvolvimento e explicação das atividades desenvolvidas na tarefa.|
| `apresentacao.pdf` | PDF da apresentação feita em aula.|
| `deploy` / `app.py` | O script principal do Streamlit. Contém a lógica para carregar os artefatos (`.pkl`) e rodar a interface web para que o usuário insira os dados e obtenha a previsão. |
| **`deploy`/`modelo_random_forest.pkl`** | **O modelo de classificação treinado.** Contém o objeto `RandomForestClassifier` final com a acurácia de 91.22\%. Utilizado para fazer a previsão do diagnóstico no Streamlit. |
| **`deploy`/`scaler_base.pkl`** | **O objeto de padronização (StandardScaler).** Armazena as médias e desvios-padrão das features de treino. Crucial para garantir que os dados de entrada do usuário sejam transformados na mesma escala em que o modelo foi treinado. |
| **`deploy`/`label_encoder.pkl`** | **O decodificador da variável alvo.** Mapeia os números preditos pelo modelo de volta para os nomes das condições médicas (e.g., 0 $\rightarrow$ 'Diabetes', 1 $\rightarrow$ 'Healthy'). |

---

## Como Rodar o Projeto

Acesse o link da aplicação em: <a href='https://trabalhoiagh.streamlit.app/'>https://trabalhoiagh.streamlit.app/</a>


Para rodar a aplicação Streamlit localmente, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/gloriaxgz/trabalho_final_IA.git
    cd trabalho_final_IA
    cd deploy
    ```

2.  **Instale as dependências:**
    (Assumindo que você tem um arquivo `requirements.txt` com `streamlit`, `pandas`, `scikit-learn`, `joblib`, etc.)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute o aplicativo Streamlit:**
    ```bash
    streamlit run app.py 
    # ou
    streamlit run streamlit_app.py 
    ```

A aplicação será aberta automaticamente no seu navegador padrão.
