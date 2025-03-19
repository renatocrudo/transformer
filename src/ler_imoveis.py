import faiss
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# carregar indice e metadados
index = faiss.read_index("imoveis_index.faiss")
with open("imoveis_metadata.json", "r") as f:
    documents = json.load(f)

# consulta do usuario (ex.: Apartamento em belenzinho abaixo de R$ 300.000)
consulta = input("Digite sua consulta: ")

embedding_consulta = model.encode(consulta, convert_to_tensor=True).astype('float32')
faiss.normalize_L2(embedding_consulta.reshape(1, -1))

# busca por 5 similaridades
k = 5
distancias, indices = index.search(embedding_consulta.reshape(1, -1), k)

# Recuperar os imoveis mais relevantes
print("\nResultados encontrados:")
print("-" * 50)
for i, idx in enumerate(indices[0]):
    resultado = documents[idx]
    print(f"\nResultado {i+1}:")
    print(f"ID: {resultado['id']}")
    print(f"Texto: {resultado['text']}")
    print(f"Similaridade: {distancias[0][i]:.4f}")
    print("-" * 50)