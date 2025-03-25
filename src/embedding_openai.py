import pandas as pd
import numpy as np
import openai
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


# nomarlizar os endereços
STREET_TYPES = [
    "al", "alameda",
    "r", "rua",
    "avenida", "av",
    "praça", "pç",
    "parque", "pq",
    "rodovia", "rod",
    "travessa", "tv"
]

TITLES = {
    "dr": "doutor",
    "dra": "doutora",
    "des": "desembargador",
    "gov": "governador",
    "cel": "coronel",
    "mal": "marechal",
    "gen": "general",
    "pref": "prefeito",
    "me": "madre",
    "pe": "padre",
    "prof": "professor",
    "profa": "professora",
    "eng": "engenheiro",
    "min": "ministro",
    "cap": "capitão",
    "maj": "major",
    "visc": "visconde",
    "srg": "sargento",
    "brig": "brigadeiro",
    "alm": "almirante",
    "pres": "presidente",
    "rep": "república",
    "sen": "senador",
    "cons": "conselheiro"
}

OTHER_STOPWORDS = [
    "das", "dos", "de", "da", "do"
]
STOPWORDS = \
    STREET_TYPES + list(TITLES.keys()) + list(TITLES.values()) + OTHER_STOPWORDS

def tokenize_addr(address: str) -> list:
    all_tokens = map(str.lower, address.split(' '))
    meaningful_tokens = []
    for token in all_tokens:
        token = token.replace('.', '').replace(',', '')
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        meaningful_tokens.append(token)
    return meaningful_tokens


def join_tokens(tokens: list) -> str:
    """Join tokens with spaces and return the normalized address string."""
    return ' '.join(tokens)

# função para pre-processar os dados
def preprocess_row(row):
    #verifico se a data de transação é datetime, senão converto para datetime
    if not isinstance(row['Data de Transação'], pd.Timestamp):
        try:
            row['Data de Transação'] = pd.to_datetime(row['Data de Transação'])
        except Exception as e:
            print(f"Erro ao converter data de transação: {e}")
            row['Data de Transação'] = datetime.now()
    
    # Normalize the address
    normalized_address = join_tokens(tokenize_addr(row['Nome do Logradouro']))
    print(f"Original: {row['Nome do Logradouro']}")
    print(f"Normalized: {normalized_address}")
        
    return f"""
        Logradouro: {normalized_address} {row['Número']}, {row['Complemento']}
        Bairro: {row['Bairro']}
        CEP: {row['CEP']}
        Valor: R$ {row['Valor de Transação (declarado pelo contribuinte)']:,.2f}
        Data: {row['Data de Transação'].strftime('%d/%m/%Y')}
        Área construída: {row['Área Construída (m2)']} m²
        Descrição IPTU: {row['Descrição do padrão (IPTU)']}
        Ano de construção: {row['ACC (IPTU)']}
    """.strip()


file_path = "dados/ITBI_2025.xlsx"
embeddings_path = "embeddings/embeddings_openai.pkl"

def load_process_data(file_path):
    print(f"Carregando dados do arquivo {file_path}...")
    df = pd.read_excel(file_path)
    # Filter only "Compra e venda" transactions
    df = df[df['Natureza de Transação'] == '1.Compra e venda']
    documents = [preprocess_row(row) for _, row in df.iterrows()]
    return documents

def generate_embeddings(documents):
    embeddings = []
    for doc in documents:
        response = openai.Embedding.create(
            model = "text-embedding-ada-002-v2",
            input = doc)
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

def save_embeddings(embeddings, documents, embeddings_path):
    with open(embeddings_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'documents': documents}, f)

def load_embeddings(embeddings_path):
    try:
        with open(embeddings_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        print(f"Erro ao carregar embeddings: {e}")
        return None, None
    
def get_most_relevant_documents(query, documents, embeddings):
    query_embedding = openai.Embedding.create(
        input = query,
        model = "text-embedding-ada-002-v2"
    )['data'][0]['embedding']

    similarities = cosine_similarity([query_embedding], embeddings)[0]
    best_match_idx = np.argmax(similarities)
    return documents[best_match_idx]

def chat_with_openai(query, documents, embeddings):
    relevant_doc = get_most_relevant_documents(query, documents, embeddings)
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = [
            {"role": "system", "content": "Você é um consultor imobiliário experiente."},
            {"role": "user", "content": f"Baseado nesse documento, responda minha pergunta: {relevant_doc}"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

def main():
    documents, embeddings = load_embeddings(embeddings_path)
    if documents is None or embeddings is None:
        documents = load_process_data(file_path)
        print(f"Processando {len(documents)} documentos...")
        print(documents[0])
        #embeddings = generate_embeddings(documents)
        #save_embeddings(embeddings, documents, embeddings_path)

    while True:
        query = input("Digite sua consulta: ")
        if query.lower() == "sair":
            break
        #response = chat_with_openai(query, documents, embeddings)
        response = "teste"
        print(f"Resposta: {response}")
    
if __name__ == "__main__":
    main()