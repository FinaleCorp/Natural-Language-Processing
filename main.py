from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')


class Answer(BaseModel):
    answer_accepted: str
    answer_given: str


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
    embeddings_ans = model.encode(answer.answer_accepted, convert_to_tensor=True)
    embeddings_student = model.encode(answer.answer_given, convert_to_tensor=True)

    similarity_score = int(float(util.cos_sim(embeddings_ans, embeddings_student)[0][0]) * 10)

    if similarity_score >= 8:
        result = "CORRECT"
    else:
        result = "INCORRECT"
    response.append(
        {
            "score": str(similarity_score),
            "result": result
        }
    )
    return {"response": response}


"uvicorn main:app --reload"