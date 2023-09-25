from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')


class Answer(BaseModel):
    answers_accepted: List[str]
    answer_given: str
    accepted_threshold: int


origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/nlp/api/v1/check")
def check_question(answer: Answer):
    response = []
    embeddings_student = model.encode(answer.answer_given, convert_to_tensor=True)

    scores = []

    for option in answer.answers_accepted:
        embeddings_ans = model.encode(option, convert_to_tensor=True)
        similarity_score = int(float(util.cos_sim(embeddings_ans, embeddings_student)[0][0]) * 10)
        scores.append(similarity_score)
    print(scores)

    scores.sort(reverse=True)
    print(scores)
    highest_similarity_score = scores[0]
    if highest_similarity_score >= answer.accepted_threshold:
        result = "CORRECT"
    else:
        result = "INCORRECT"
    response.append(
        {
            "score": str(highest_similarity_score),
            "result": result
        }
    )
    return {"response": response}


"uvicorn main:app --reload"