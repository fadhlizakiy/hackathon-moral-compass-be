import re
from numpy import lib
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from typing import Union, List
from pydantic import BaseModel
from fastapi import FastAPI
import numpy as np 
from numpy.linalg import norm
import csv

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = CountVectorizer()
libQuestions  = []
libAnswersIdx = []
libAnswers    = []

with open('question.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            libQuestions.append(row[1])
            line_count += 1
vectorizer.fit(libQuestions)
vectorLibQuestions = vectorizer.transform(libQuestions)
arrVectorLibQuestions = vectorLibQuestions.toarray()

with open('answer.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            libAnswersIdx.append(int(row[1]))
            libAnswers.append(row[2])
            line_count += 1

# Build Decision Model
class Decision(BaseModel):
    question: str
    options: List[str] = []

class Question(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/decide")
async def count_decision(req: Decision):
    # variable for storing the final options
    options = []

    # sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    # for loop for each options
    for i in req.options:
        sentiment = sentiment_analyzer.polarity_scores(i)
        # put the option as a result
        option_result = {
            "option":i, 
            "neg":sentiment['neg'], 
            "neu":sentiment['neu'], 
            "pos":sentiment['pos'], 
            "compound" : sentiment['compound']}

        # append to the options variable for final return
        options.append(option_result)

    # make vector of the question
    vectorQuestion = vectorizer.transform([req.question])
    arrVectorQuestion = vectorQuestion.toarray()

    # find cosine similarity
    cosine = np.sum(arrVectorLibQuestions*arrVectorQuestion, axis=1)/(norm(arrVectorLibQuestions, axis=1)*norm(arrVectorQuestion, axis=1))
    max_val = max(cosine)
    idx = int(np.where(cosine == max_val)[0][0]) + 1

    # find corresponding pre-trained question
    idy = np.where(np.array(libAnswersIdx) == idx)[0]
    results = []
    for row in idy:
        results.append(libAnswers[row])

    return {"question": req.question, "options": options, "additional" : results}

@app.get("/data/question")
async def get_question_data():
    questions = []
    with open('question.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                questions.append({"id":row[0], "question":row[1], "domain" : row[2], "frequency": row[3]})
                line_count += 1
    return {"questions" : questions}

@app.post("/test/question")
async def test_question_data(req: Question):
    
    vectorQuestion = vectorizer.transform([req.question])
    arrVectorQuestion = vectorQuestion.toarray()
    
    cosine = np.sum(arrVectorLibQuestions*arrVectorQuestion, axis=1)/(norm(arrVectorLibQuestions, axis=1)*norm(arrVectorQuestion, axis=1))
    max_val = max(cosine)
    idx = int(np.where(cosine == max_val)[0][0]) + 1

    return {"id": idx, "value" : max_val}

@app.post("/test/question-answer")
async def test_question_data(req: Question):
    # make vector of the question
    vectorQuestion = vectorizer.transform([req.question])
    arrVectorQuestion = vectorQuestion.toarray()

    # find cosine similarity
    cosine = np.sum(arrVectorLibQuestions*arrVectorQuestion, axis=1)/(norm(arrVectorLibQuestions, axis=1)*norm(arrVectorQuestion, axis=1))
    max_val = max(cosine)
    idx = int(np.where(cosine == max_val)[0][0]) + 1

    # find corresponding pre-trained question
    idy = np.where(np.array(libAnswersIdx) == idx)[0]
    results = []
    for row in idy:
        results.append(libAnswers[row])

    return {"question": req.question, "additional" : results}

@app.post("/test/question-answer-final")
async def test_question_data(req: Decision):
    # variable for storing the final options
    options = []

    # sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    # for loop for each options
    for i in req.options:
        sentiment = sentiment_analyzer.polarity_scores(i)
        # put the option as a result
        option_result = {
            "option":i, 
            "neg":sentiment['neg'], 
            "neu":sentiment['neu'], 
            "pos":sentiment['pos'], 
            "compound" : sentiment['compound']}

        # append to the options variable for final return
        options.append(option_result)

    # make vector of the question
    vectorQuestion = vectorizer.transform([req.question])
    arrVectorQuestion = vectorQuestion.toarray()

    # find cosine similarity
    cosine = np.sum(arrVectorLibQuestions*arrVectorQuestion, axis=1)/(norm(arrVectorLibQuestions, axis=1)*norm(arrVectorQuestion, axis=1))
    max_val = max(cosine)
    idx = int(np.where(cosine == max_val)[0][0]) + 1

    # find corresponding pre-trained question
    idy = np.where(np.array(libAnswersIdx) == idx)[0]
    results = []
    for row in idy:
        results.append(libAnswers[row])

    return {"question": req.question, "options": options, "additional" : results}

