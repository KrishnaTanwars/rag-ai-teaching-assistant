import os
import json
import numpy as np
import pandas as pd
import joblib
import requests


def create_embedding(text_list):
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "bge-m3",
                "input": text_list
            }
        )

        response = r.json()

        if "embeddings" not in response:
            print("Embedding error:", response)
            return None

        embeddings = [np.array(e) for e in response["embeddings"]]
        return embeddings

    except Exception as e:
        print("Embedding failed:", e)
        return None


# If embeddings already exist → load
if os.path.exists("embeddings.joblib"):
    print("Embeddings already exist. Loading...")
    df = joblib.load("embeddings.joblib")

else:
    jsons = os.listdir("newjsons")

    my_dicts = []
    chunk_id = 0

    for json_file in jsons:

        with open(f"newjsons/{json_file}", encoding="utf-8") as f:
            content = json.load(f)

        print(f"Creating Embeddings for {json_file}")

        texts = [str(c.get("text","")).strip() for c in content["chunks"] if str(c.get("text","")).strip()]

        embeddings = create_embedding(texts)

        if embeddings is None:
            print("Embedding generation failed.")
            exit()

        for i, chunk in enumerate(content["chunks"]):

            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embeddings[i]

            chunk_id += 1
            my_dicts.append(chunk)

    df = pd.DataFrame.from_records(my_dicts)

    # Save embeddings
    joblib.dump(df, "embeddings.joblib")

    print("Embeddings saved to embeddings.joblib")