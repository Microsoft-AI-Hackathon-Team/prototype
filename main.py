import json
import os

from langchain.docstore.document import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "lm-studio"

"""
def load_json_file():
    docs = list()
    with open("../scraper/courses.json", "r") as file:
        data = json.load(file)

    for course in data["courses"]:
        content = ""
        for key, value in course.items():
            content += f"{key}: {value}\n"
        docs.append(Document(page_content=content))

    return docs


# loader = JSONLoader(file_path="../scraper/courses.json", jq_schema=".courses[]")
# docs = loader.load()
docs = load_json_file()

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1",
    check_embedding_ctx_length=False,
    model="nomic-ai/nomic-embed-text-v1.5-GGUF",
)


# Split docs into smaller chunks to fit SQLite Query
def split_docs(docs):
    chunk_size = 100
    chunks = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]
    return chunks


docs_chunked = split_docs(docs)

for split in docs_chunked:
    vectorstore = Chroma.from_documents(
        split,
        embeddings,
        persist_directory="./chroma_db",
        collection_name="course_descriptions",
    )

# process grades csv
loader = CSVLoader(file_path="../scraper/2023_msu_grades.csv")
docs = loader.load()
docs_chunked = split_docs(docs)

for split in docs_chunked:
    vectorstore = Chroma.from_documents(
        split,
        embeddings,
        persist_directory="./chroma_db",
        collection_name="grades",
    )
""" ""

# embeddings = OpenAIEmbeddings(
#    base_url="http://localhost:1234/v1",
#    check_embedding_ctx_length=False,
#    model="nomic-ai/nomic-embed-text-v1.5-GGUF",
# )

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


# llm = ChatOpenAI(
#    base_url="http://localhost:1234/v1",
#    api_key="lm-studio",
#    model="TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q8_0.gguf",
# )

llm = ChatOpenAI(model="gpt-4o")

PROMPT_TEMPLATE = """[INST]You are an AI assistant, and provides academic advising to students at Michigan State University by using data from MSU grades dataset, MSU course descriptions dataset. You can answer questions about courses, professors, and grades. If you don't know, just say "I don't know", don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or number when possible.[/INST]
[Assistant]"""


retriever = vectorstore.as_retriever()


def query_rag(query: str):
    db_cd = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="course_descriptions",
    )

    results = db_cd.similarity_search_with_relevance_scores(query, k=5)

    print(f"Found {len(results)} results")
    if len(results) == 0 or results[0][1] < 0.5:
        print("I don't know")
        exit(1)

    context_text = "\n\n".join([doc.page_content for doc, _score in results])

    db_grades = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="grades",
    )

    results = db_grades.similarity_search_with_relevance_scores(query, k=20)
    print(f"Found {len(results)} results")

    context_text += "\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    response_text = llm.invoke(prompt).content

    sources = [doc.metadata.get("source", None) for doc, _score in results]

    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    return formatted_response, response_text


# query = "Should I take CSE 232 along with CSE 260? Why, why not?"
# formatted_response, response_text = query_rag(query)
# print(response_text)


import gradio as gr
from langchain.schema import AIMessage, HumanMessage


def predict(message, history):

    history_langchain_format = list()
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    _, response_text = query_rag(message)
    return response_text


gr.ChatInterface(predict).launch()
