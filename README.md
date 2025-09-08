# 🍕 Case Técnico Data Science – iFood

Este repositório contém a solução desenvolvida para o **Case Técnico de Data Science do iFood**, cujo objetivo é realizar um **estudo baseado em dados** sobre a distribuição de cupons e ofertas aos clientes. 

---

## 📖 Descrição do Case

O desafio envolve:
1. **Analisar dados históricos** de clientes, ofertas e transações.  
2. **Desenvolver um modelo/técnica** para decidir qual oferta enviar a cada cliente.  
3. **Comunicar resultados de forma clara**, com estimativas do impacto para o negócio.  

📊 **Dados fornecidos**:
- `offers.json`: informações sobre tipos de ofertas, valor mínimo, desconto, duração e canais.  
- `profile.json`: atributos de ~17k clientes (idade, gênero, limite de crédito, data de registro).  
- `transactions.json`: ~300k eventos (transações, ofertas recebidas, visualizadas ou concluídas).  

---

## ⚙️ Instalação e Configuração do Ambiente (Linux)

### 1. Pré-requisitos

- **Linux (Ubuntu/Debian recomendado)**
- **Python 3.11+**
- **Java JDK 11 (LTS)** → necessário para rodar o PySpark

Instale com:

```bash
sudo apt update && sudo apt install openjdk-11-jdk -y
java -version
´´´

1. Pré-requisitos. Criar ambiente virtual  

Com venv:

python3.11 -m venv venv
source venv/bin/activate

3. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt


📌 Principais pacotes:

pyspark==4.0.0 → processamento distribuído

scikit-learn==1.7.1 → modelagem e métricas

matplotlib, seaborn, missingno → visualizações

pyarrow → integração com dados colunares

tqdm → barras de progresso

ipykernel → execução dos notebooks

4. Adicionar kernel no Jupyter

python -m ipykernel install --user --name=venv --display-name "py3.11_case_ifood"


📂 Estrutura do Projeto
ifood-case/
├── data/                 # Datasets
│   ├── raw/              # Dados originais
│   └── processed/        # Dados processados (bronze, silver, gold)
├── models/               # Modelo ML salvo
├── notebooks/            # Jupyter notebooks
│   ├── 1_data_processing.ipynb   # Limpeza e unificação dos dados
│   └── 2_modeling.ipynb          # Modelagem e avaliação
├── presentation/         # Slides executivos
├── src/                  # Código auxiliar (funções utilitárias)
│   └── __init__.py       # Utilizado como módulo
│   └── aux.py            # Funções gerais
│   └── plot_functions.py # Funções de visualização
│   └── projeto.py        # Funções para projeto (notebook 2)
├── .gitignore            # ignore
├── LICENSE               # Creative Commons Legal Code
├── README.md             # Este arquivo
└── requirements.txt      # Dependências


👤 Autor

Julio Patti Pereira
📧 juliopatti@gmail.com