import requests
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_json import create_embedding


def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    response = r.json()
    return response["response"]

# def inference_openai(prompt):
#     response = client.responses.create(
#         model = "gpt-5.4",
#         input = prompt
#     )
#     return response.choices[0]


df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]


similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()
top_results = 5
max_idx = similarities.argsort()[::-1][0:top_results]
# print(max_idx)
new_df = df.loc[max_idx]
# print(new_df[["title", "num","text"]])


prompt = f"""
You are helping me find where a topic is explained in my YouTube videos.

Below are subtitle chunks from my videos in JSON format:
{new_df[['title','number','start','end','text']].to_json(orient="records")}

User Question:
{incoming_query}

Instructions:
Talk like a helpful friend and give a very clear answer.

Rules:
1. Identify the MOST relevant video.
2. Mention the exact timestamp.
3. Briefly explain what is discussed there in 1–2 lines.
4. Do NOT list multiple unrelated videos.
5. If nothing matches well, say you couldn't find it in the videos.

Answer format:

Video: <video title>
Video Number: <number>
Timestamp: <start sec> - <end sec>

Explanation:
<short friendly explanation>
"""

with open("prompt.txt", "w") as f:
    f.write(prompt)

# response = inference(prompt)["response"]
# print(response)

response = inference(prompt)

with open("response.txt", "w") as f:
    print("\nAnswer:\n")
    print(response)
    f.write(response)
# for index,item in new_df.iterrows():
#     print(index, item["title"], item["num"], item["text"],item["start"], item["end"])