from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')


class Answer(BaseModel):
    answersAccepted: List[str]
    answersGiven: List[str]
    acceptedThreshold: int


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


class Score:
    answer: str
    score: int
    result: str


@app.post("/nlp/api/v1/check")
def check_question(answer: Answer):
    response = []

    total_scores = []

    for answerObj in answer.answersGiven:
        embeddings_student = model.encode(answerObj, convert_to_tensor=True)
        scores = []
        for option in answer.answersAccepted:
            embeddings_ans = model.encode(option, convert_to_tensor=True)
            similarity_score = int(float(util.cos_sim(embeddings_ans, embeddings_student)[0][0]) * 10)
            if similarity_score >= answer.acceptedThreshold:
                answer.answersAccepted.remove(option)
            scores.append(similarity_score)

        scores.sort(reverse=True)
        print(scores)
        score_obj = Score()
        score_obj.score = scores[0]
        if scores[0] >= answer.acceptedThreshold:
            score_obj.result = "CORRECT"
        else:
            score_obj.result = "INCORRECT"

        score_obj.answer = answerObj
        total_scores.append(score_obj)

    return {"response": total_scores}

"uvicorn main:app --reload"
