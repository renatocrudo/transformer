import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import faiss
from datetime import datetime
import argparse
from sentence_transformers import SentenceTransformer
# função para pre-processar os dados
def preprocess_row(row):
    #verifico se a data de transação é datetime, senão converto para datetime
    if not isinstance(row['Data de Transação'], pd.Timestamp):
        try:
            row['Data de Transação'] = pd.to_datetime(row['Data de Transação'])
        except Exception as e:
            print(f"Erro ao converter data de transação: {e}")
            row['Data de Transação'] = datetime.now()
        
    return f"""
        Logradouro: {row['Nome do Logradouro']} {row['Número']}, {row['Complemento']}
        Bairro: {row['Bairro']}
        CEP: {row['CEP']}
        Valor: R$ {row['Valor de Transação (declarado pelo contribuinte)']:,.2f}
        Data: {row['Data de Transação'].strftime('%d/%m/%Y')}
        Área construída: {row['Área Construída (m2)']} m²
        Descrição IPTU: {row['Descrição do padrão (IPTU)']}
        Ano de construção: {row['ACC (IPTU)']}
        Matricula do imóvel: {row['Matrícula do Imóvel']}
        Número de cadastro: {row['N° do Cadastro (SQL)']}
    """.strip()

# função para carregar os dados salvos no FAISS
def load_saved_data(index_path, mapping_path):
    #carregando o índice FAISS salvo
    index = faiss.read_index("faiss_index.bin")
    print("FAISS index carregado com sucesso!")

    # Carregando o CVS com os dados originais
    df = pd.read_csv("documents_mapping.csv")
    # convertendo a coluna data para datetime
    if 'Data de Transação' in df.columns:
        df['Data de Transação'] = pd.to_datetime(df['Data de Transação'], errors='coerce')
    
    return index, df

def build_index(path_xlsx="dados/ITBI_2025.xlsx", index_path="faiss_index.bin", mapping_path="documents_mapping.csv"):
    try:
        df = pd.read_excel(path_xlsx)
        print("Arquivo xlsx carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o arquivo xlsx: {e}")
        return
    
    if not pd.api.types.is_datetime64_any_dtype(df['Data de Transação']):
        df['Data de Transação'] = pd.to_datetime(df['Data de Transação'], errors='coerce')

    # Filter only "Compra e venda" transactions
    df = df[df['Natureza de Transação'] == '1.Compra e venda']
    print(f"Filtrando apenas transações de compra e venda. Total de documentos: {len(df)}")

    # preprocessando os dados
    documents = df.apply(preprocess_row, axis=1).tolist()
    print(f"Preprocessando {len(documents)} documentos...")

    #Carrega o modelo de embeddings
    #model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    #embed = hub.load(model_url)
    print("Carregando o modelo Sentence Transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Modelo de embeddings carregado com sucesso!")
    
    #gerando os embeddings dos documentos
    #embeddings = embed(documents).numpy()
    embeddings = model.encode(documents, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    print("Embeddings gerados com sucesso com shape:", embeddings.shape)

    #criando o índice FAISS
    dim = embeddings.shape[768] # dimensão dos embeddings
    #index = faiss.IndexFlatL2(dim) # criando o índice com a métrica de distância euclidiana
    index = faiss.IndexFlatIP(dim) # criando o índice com a métrica de produto interno
    index.add(embeddings) # adicionando os embeddings ao índice
    print(f"Índice FAISS criado com {index.ntotal} documentos")

    #salvando o índice FAISS e o mapeamento de documentos
    faiss.write_index(index, index_path)
    df.to_csv(mapping_path, index=False)
    print("Índice FAISS e mapeamento de documentos salvos com sucesso!")
    


# Função que recupera os dados do FAISS
def retrieve_documents(query, index, df, model, top_k=5):
    # Gera o embedding da query com o mesmo modelo
    #query_embedding = embed([query]).numpy()
    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # utilizando p Faiss para encontrar os top_k resultados mais relevantes
    top_k = 5 #index.ntotal
    distances, indices = index.search(query_embedding, top_k)



    #armazendo os resultados recuperados
    retrieved_docs = []
    # definir threshold para considerar um documento relevante
    similarity_threshold = 0.8
    for idx, distance in zip(indices[0], distances[0]):
        if distance <= similarity_threshold:
            # formata o documento utilizando a função preprocess_row
            doc_text = preprocess_row(df.iloc[idx])
            retrieved_docs.append(doc_text)
    
    return retrieved_docs


# Função para gerar um prompt que combina os documentos recuperados com a query do usuário
def generate_chatgpt_prompt(query, context_docs):
    context_text = "\n\n".join(context_docs)
    prompt = f"""
Utilize as informações abaixo para responder a pergunta de forma detalhada:

Contexto:
{context_text}

Pergunta: {query}

Resposta:
"""
    return prompt

def query_index(query, top_k, index_path="faiss_index.bin", mapping_path="documents_mapping.csv"):
    # carregando o indice faiss
    index, df = load_saved_data(index_path, mapping_path)

    # carregando o modelo de embeddings
    #model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    
    print("Carregando o modelo Sentence Transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #embed = hub.load(model_url)
    print("Modelo de embeddings carregado com sucesso!")

    #recuperando os documentos mais relevantes
    retrieved_docs = retrieve_documents(query, index, df, model, top_k=top_k)
    print(f"{len(retrieved_docs)} documentos recuperados:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\nDocumento {i}:\n{'-'*50}\n{doc}")  

    # gerando a resposta do chatgpt
    prompt = generate_chatgpt_prompt(query, retrieved_docs)

    print("\nPrompt para o ChatGPT:")
    print(prompt)


# Função principal que utiliza argumentos de linha de comando para escolher a ação desejada
def main():
    parser = argparse.ArgumentParser(description="Script para construir índice FAISS a partir de um XLSX e realizar consultas com embeddings.")
    parser.add_argument("--build", action="store_true", help="Construir o índice FAISS a partir do arquivo XLSX")
    parser.add_argument("--query", action="store_true", help="Realizar uma consulta utilizando o índice FAISS salvo")
    parser.add_argument("--q", type=str, help="Texto da query para consulta")
    parser.add_argument("--xlsx", type=str, default="dados/ITBI_2025.xlsx", help="Caminho para o arquivo XLSX")
    parser.add_argument("--index", type=str, default="faiss_index.bin", help="Caminho para salvar/carregar o índice FAISS")
    parser.add_argument("--mapping", type=str, default="documents_mapping.csv", help="Caminho para salvar/carregar o mapeamento dos documentos")
    parser.add_argument("--topk", type=int, default=3, help="Número de documentos a recuperar")
    args = parser.parse_args()

    if args.build:
        print("Construindo o índice FAISS...")
        #build_index(xlsx_path=args.xlsx, index_path=args.index, mapping_path=args.mapping)
        build_index(path_xlsx="dados/ITBI_2025.xlsx")
    elif args.query:
        if not args.q:
            print("Por favor, forneça uma query utilizando o argumento --q")
        else:
            print("Realizando consulta no índice FAISS...")
            #query_index(query=args.q, top_k=args.topk, index_path=args.index, mapping_path=args.mapping)
            query_index(query=args.q, top_k=5, index_path="faiss_index.bin", mapping_path="documents_mapping.csv")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()