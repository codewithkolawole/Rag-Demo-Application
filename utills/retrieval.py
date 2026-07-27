import os.path
import faiss
import numpy as np
import pickle
from .chunking import chunk_text
from .embedding import data_embed



def load_faiss_index():
    index_path = "faiss_store/index.faiss"
    mapping_path= "faiss_store/chunk_map.pkl"

    valid = os.path.exists(index_path) and os.path.exists(mapping_path) >0
    valid = valid and os.path.exists(mapping_path) and os.path.getsize(mapping_path) > 0

    if valid:
        try:
            index = faiss.read_index(index_path)
            with open(mapping_path,"rb") as f:
                chunk_map = pickle.load(f)
            return index, chunk_map
        except Exception as e:
            print("Corrupted index file detected", e)

    print("Generating new index from the file")

    with open("data/kolawole.txt","r",encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    all_embeddings = []
    chunk_map = []
    for chunk in chunks:
        emb = data_embed(chunk)
        all_embeddings.append(emb)
        chunk_map.append(chunk)

    all_embeddings = np.array(all_embeddings).astype("float32")
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    os.makedirs("faiss_store", exist_ok=True)
    faiss.write_index(index, "faiss_store/index.faiss")
    with open("faiss_store/chunk_map.pkl", "wb") as f:
        pickle.dump(chunk_map, f)
    return index, chunk_map


def retrieve_top_k(query,index,chunk_map, k=3):
    query_emb = data_embed(query)
    distances, indices = index.search(np.array([query_emb]).astype("float32"), k)
    return [chunk_map[i] for i in indices[0]]