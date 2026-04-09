# TCC: Aplicação de Algoritmos de Machine Learning para Análise de Sentimentos no Contexto Educacional

Análise de sentimentos em reviews de aplicativos educacionais comparando métodos lexicais, Machine Learning clássico e BERT.

---

##  Abordagens Utilizadas

- Métodos Léxicos (SentiLex)
- Machine Learning com TF-IDF
  - Naive Bayes
  - SVM
  - Regressão Logística
  - Random Forest
  - Gradient Boosting
- Machine Learning com embeddings do BERT
  - Naive Bayes
  - SVM
  - Regressão Logística
  - Random Forest
  - Gradient Boosting
- Fine-tuning do BERT (completo e parcial)

## Avaliação

Os modelos foram avaliados utilizando:
- Acurácia
- Precisão
- Recall
- F1-score
- Matriz de confusão
- Interpretabilidade com LIME

---

## Instalação Rápida

```bash
# 1. Criar ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Baixar modelos NLP
python -m spacy download pt_core_news_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('rslp')"

# 4. Criar estrutura de pastas
mkdir -p data/{raw,processed,lexicons} results/{figures,models,metrics}

# 5. Baixar SentiLex
# Coloque SentiLex-lem-PT02.txt em data/lexicons/
```

---

## 📁 Estrutura

```
TCC_Analise_Sentimentos/
├── notebooks/           # 7 notebooks (execute em ordem)
├── data/
│   ├── raw/            # Dados brutos (gerado)
│   ├── processed/      # Dados processados (gerado)
│   └── lexicons/       # SentiLex
└── results/
    ├── figures/        # Gráficos (gerado)
    ├── models/         # Modelos (gerado)
    └── metrics/        # Métricas CSV (gerado)
```

---

## ▶️ Execução

**Execute os notebooks NA ORDEM:**

```
01_Coleta_Dados.ipynb              
02_ETL_e_AED.ipynb                 
03_PLN.ipynb                      
04_TF_IDF_5_Modelos.ipynb           
05_Metodos_Lexicais.ipynb          
06_BERT_Embeddings_5_Modelos.ipynb 
07_BERT_Fine_Tuning.ipynb         
```

---

## Descrição dos Notebooks

| # | Notebook | O que faz |
|---|----------|-----------|
| 1 | Coleta | Coleta reviews Google Play (Duolingo, Khan, etc) |
| 2 | ETL/AED | Rotula sentimentos, balanceia, análise exploratória |
| 3 | PLN | Tokeniza, remove stopwords, stemming, lematização |
| 4 | TF-IDF | Vetoriza TF-IDF + treina 5 modelos ML + LIME |
| 5 | Lexical | Métodos baseados em SentiLex (baseline) |
| 6 | BERT Emb | BERT extrai embeddings + treina 5 modelos ML |
| 7 | BERT FT | Fine-Tuning completo do BERT |

---

## Tecnologias

**PLN:** NLTK, spaCy  
**ML:** scikit-learn (NB, SVM, RF, LR, GB)  
**Deep Learning:** PyTorch, Transformers (BERTimbau)  
**Interpretabilidade:** LIME
**Dados:** pandas, numpy  
**Visualização:** matplotlib, seaborn, wordcloud  

---

## ⚠️ Troubleshooting

**Erro spaCy:**
```bash
pip install https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.7.0/pt_core_news_sm-3.7.0-py3-none-any.whl
```

**Erro accelerate (Fine-Tuning):**
```bash
pip install accelerate>=0.25.0
# Depois reinicie o kernel
```

**Erro pyarrow:**
```bash
pip install pyarrow==14.0.2
```

**Out of Memory (BERT):**
```python
# Reduza batch_size no notebook
per_device_train_batch_size=8  # era 16
```

# 📥 Download Necessário
## SentiLex (Léxico PT-BR):

- Arquivo: SentiLex-lem-PT02.txt
- Baixar: https://github.com/paulagd/SentiLex-PT
- Salvar em: data/lexicons/SentiLex-lem-PT02.txt

---

## 📞 Observações

- Execute notebooks **sequencialmente** (cada um depende do anterior)
- Kernels devem usar o ambiente `venv`
- Todos os dados são gerados automaticamente (exceto SentiLex)

---


**Desenvolvido por Isabella Silva Marcondes**
