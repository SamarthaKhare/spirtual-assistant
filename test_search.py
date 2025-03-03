import pandas as pd
import numpy as np
from ast import literal_eval
from utils_v1.get_gemini_embeddings import get_embedding
from scipy.spatial.distance import cosine

def ensure_list(embedding):
    if isinstance(embedding, str):
        try:
            return literal_eval(embedding)
        except:
            return None
    return embedding

def safe_cosine_similarity(x, y):
    if x is None or y is None:
        return 0.0
    try:
        return 1 - cosine(np.array(x), np.array(y))  # Convert cosine distance to similarity
    except Exception as e:
        print(f"Error in cosine similarity calculation: {e}")
        return 0.0

def search_functions(df, code_query, n=3):
    from sklearn.preprocessing import normalize
    query_embedding = get_embedding(code_query)
    df['code_embedding'] = df['code_embedding'].apply(ensure_list)
    df = df[df['code_embedding'].notnull()]
    df['similarities'] = df['code_embedding'].apply(lambda x: safe_cosine_similarity(x, query_embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res

