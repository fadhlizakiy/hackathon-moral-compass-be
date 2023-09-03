import random
from _operator import itemgetter

from typing import Dict, List

import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever


import multiprocessing


# Process class
class Process(multiprocessing.Process):
    def __init__(self, _id, task, task_input, result_holder):
        super(Process, self).__init__()
        self.id = _id
        self.task = task
        self.task_input = task_input
        self.result_holder = result_holder

    def insert_result(self, data):
        self.result_holder.append(data)

    def run(self):
        print("I'm the process with id: {}".format(self.id))
        data = self.task(**self.task_input)
        self.insert_result(data=data)


def get_api_key():
    return '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'

def get_functions_for_output():
    functions = [
        {
            "name": "ethical_answer",
            "description": "An ethical answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "short_answer": {
                        "type": "string",
                        "description": "Clear answer for question in single sentence with brief explanation"
                    },
                    "ethical_score": {
                        "type": "string",
                        "description": "Score of answer on scale from provided in prompt"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Fairly detailed explanation for providing such an answer"
                    },
                    "question": {
                        "type": "string",
                        "description": "Exact question that is being answered"
                    },
                },
                "required": ["short_answer", "explanation", "ethical_score"]
            }
        }
    ]

    return functions


def document_loader(document_name: str, content_column_name: str, localization: str | None, lazy_load: bool = False):
    name = f"{localization}_{document_name}" if localization else document_name
    df = pd.read_csv(filepath_or_buffer=f"./{name}")
    loader = DataFrameLoader(df, page_content_column=content_column_name)

    result = loader.load()

    return result


def get_vector_store(data_input, data_type) -> Chroma | None:
    match data_type:
        case "doc":
            return Chroma.from_documents(data_input, OpenAIEmbeddings(openai_api_key=get_api_key()))
        case _:
            return None


def get_retriever(localization: str | None = None) -> VectorStoreRetriever:
    # Create the retriever from the question
    _dl = document_loader(document_name="answer.csv", localization=localization, content_column_name="answer")
    _vs = get_vector_store(data_input=_dl, data_type="doc")

    retriever = _vs.as_retriever()

    return retriever


def transform_prompt_data(prompt_input: Dict) -> Dict:
    result: Dict = {}
    template: str = ""

    localization = prompt_input.get("localization", None)

    # get context
    template += (
        """Answer the question based you understanding of issue while """
        """trying to base your decision on the following context:\n {context}"""
        """\nIf context does not make it possible for you to give answer """
        """you can ignore it and use your own assumption."""
    )
    result["context"] = itemgetter("question") | get_retriever(localization=localization)

    # localization
    if localization:
        template += """\nAssume issue is happening in country:\n {localization}"""
        result["localization"] = itemgetter("localization")

    # question
    template += """\nQuestion:\n {question}"""
    result["question"] = itemgetter("question")

    # scale
    template += (
        """\nAnswer has to be followed by how confident are you that answer is ethical on scale:\n {scale}. """
        """Bottom of scale being not ethical at all, while top of scale is completely ethical."""
    )
    result["scale"] = itemgetter("scale")

    # # amount_of_options
    # template += (
    #     """\nDo your best to provide multiple answers that are provided in array, amount being:\n {amount_of_options}"""
    # )
    # result["amount_of_options"] = itemgetter("amount_of_options")
    return {
        "template": template,
        "chain": result,
    }


def get_answer(prompt_input: Dict[str, str]):
    model = ChatOpenAI(openai_api_key=get_api_key())
    output_functions = get_functions_for_output()

    model = model.bind(function_call={"name": "ethical_answer"}, functions=output_functions) if output_functions else model
    chain_data = transform_prompt_data(prompt_input=prompt_input)

    chain = (
        chain_data["chain"]
        | ChatPromptTemplate.from_template(chain_data["template"])
        | model
    )
    chain = chain | StrOutputParser() if not output_functions else chain

    result = chain.invoke(prompt_input)

    return result


def search_similar_entries(db, question, amount):
    docs = db.similarity_search(question)
    return [doc.page_content for doc in docs[:amount]]


def get_user_input() -> str:
    choice_field: List[str] = [
        "Is it ethical to fire someone for their poor performance?",
        "Is it ethical to reward only one of team members?",
        "Is it ethical to be entering conflict for personal gains?",
        "is it okay to be angry for ceremonies happening so often?"
    ]
    return random.choice(
        choice_field
    )


if __name__ == '__main__':

    amount_of_options = 3
    random_input = {
        "localization": "indonesia",
        "question": get_user_input(),
        "scale": "-10,10",
    }
    a = get_answer(prompt_input=random_input)
    print(a)
