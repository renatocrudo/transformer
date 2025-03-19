import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import warnings
import json
import torch
from datetime import datetime

warnings.filterwarnings("ignore", category=ResourceWarning)

# Custom JSON encoder to handle Timestamp objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

df = pd.read_excel("dados/ITBI_2025.xlsx", sheet_name="JAN-2025")

   # Criando um pre-processamento para ter um texto semi-estruturado
def preprocess_row(row):
    return f"""
        Logradouro: {row['Nome do Logradouro']} {row['Número']}, {row['Complemento']}
        Bairro: {row['Bairro']}
        CEP: {row['CEP']}
        Valor: R$ {row['Valor de Transação (declarado pelo contribuinte)']:,.2f}
        Data: {row['Data de Transação'].strftime('%d/%m/%Y')}
        Área construída: {row['Área Construída (m2)']} m²
        Descrição IPTU: {row['Descrição do padrão (IPTU)']}
        Ano de construção: {row['ACC (IPTU)']}
        Matrícula do imóvel: {row['Matrícula do Imóvel']}
        Número de cadastro: {row['N° do Cadastro (SQL)']}
        """.strip()

# Convert DataFrame rows to dictionaries with proper datetime handling
documents = []
for idx, row in df.iterrows():
    metadata = row.to_dict()
    # Convert Timestamp objects to strings in metadata
    for key, value in metadata.items():
        if isinstance(value, pd.Timestamp):
            metadata[key] = value.strftime('%Y-%m-%d %H:%M:%S')
    
    documents.append({
        "id": idx,
        "text": preprocess_row(row),
        "metadata": metadata
    })

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_np = np.array([model.encode(doc['text'], convert_to_tensor=False) for doc in documents]).astype('float32')

faiss.normalize_L2(embeddings_np)

index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)

faiss.write_index(index, "imoveis_index.faiss")
del index

with open("imoveis_metadata.json", "w") as f:
    json.dump(documents, f, cls=DateTimeEncoder)

# Clean up CUDA resources if available
#if torch.cuda.is_available():
#    torch.cuda.empty_cache()

print("Embeddings and metadata saved!")