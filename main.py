import json
import os
from langchain.docstore.document import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import gradio as gr
from langchain.schema import AIMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not set in environment variables")

# Initialize embeddings with correct API key
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Directory for Chroma persistence
persist_directory = "./chroma_db"

# # Function to load JSON data
# def load_json_file(file_path):
#     docs = []
#     with open(file_path, "r") as file:
#         data = json.load(file)
        
#         if "courses.json" in file_path:
#             for item in data["courses"]:
#                 content = ""
#                 for key, value in item.items():
#                     content += f"{key}: {value}\n"
#                 docs.append(Document(page_content=content))

#         elif "organizations.json" in file_path:
#             for item in data["organizations"]:
#                 content = ""
#                 for key, value in item.items():
#                     content += f"{key}: {value}\n"
#                 docs.append(Document(page_content=content))
#     return docs

# # Load course descriptions
# course_docs = load_json_file("../scraper/courses.json")

# # Load organizations
# org_docs = load_json_file("../scraper/organizations.json")

# # Function to split documents into chunks
# def split_docs(docs, chunk_size=100):
#     chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
#     return chunks

# Check if embeddings already exist to avoid recomputation
# if not os.path.exists(persist_directory):
#     os.makedirs(persist_directory)

# Check if specific collections exist
# course_descriptions_path = os.path.join(persist_directory, 'course_descriptions')
# organizations_path = os.path.join(persist_directory, 'organizations')
# grades_path = os.path.join(persist_directory, 'grades')

# # Create course descriptions embeddings if not exist
# if not os.path.exists(course_descriptions_path):
#     course_docs_chunked = split_docs(course_docs)
#     for split in course_docs_chunked:
#         vectorstore = Chroma.from_documents(
#             split,
#             embeddings,
#             persist_directory=persist_directory,
#             collection_name="course_descriptions",
#         )

# # Create organizations embeddings if not exist
# if not os.path.exists(organizations_path):
#     org_docs_chunked = split_docs(org_docs)
#     for split in org_docs_chunked:
#         vectorstore = Chroma.from_documents(
#             split,
#             embeddings,
#             persist_directory=persist_directory,
#             collection_name="organizations",
#         )

# # Create grades embeddings if not exist
# if not os.path.exists(grades_path):
#     loader = CSVLoader(file_path="../scraper/2023_msu_grades.csv")
#     grade_docs = loader.load()
#     grade_docs_chunked = split_docs(grade_docs)
#     for split in grade_docs_chunked:
#         vectorstore = Chroma.from_documents(
#             split,
#             embeddings,
#             persist_directory=persist_directory,
#             collection_name="grades",
#         )

# Initialize LLM
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")

PROMPT_TEMPLATE = """[INST]You are an AI assistant, and provide academic advising to students at Michigan State University by using data from MSU grades dataset, MSU course descriptions dataset, and student organizations dataset. You can answer questions about courses, professors, grades, and student clubs/organizations. If you don't know, just say "I don't know", don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.[/INST]
[Assistant]"""

# Function to query the vectorstore and LLM
def query_rag(query: str):
    # Initialize vectorstores
    db_cd = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="course_descriptions",
    )
    db_grades = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="grades",
    )
    db_orgs = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="organizations",
    )

    # Search course descriptions
    results_cd = db_cd.similarity_search_with_relevance_scores(query, k=5)
    context_text_cd = "\n\n".join([doc.page_content for doc, _ in results_cd])

    # Search grades
    results_grades = db_grades.similarity_search_with_relevance_scores(query, k=5)
    context_text_grades = "\n\n".join([doc.page_content for doc, _ in results_grades])

    # Search organizations
    results_orgs = db_orgs.similarity_search_with_relevance_scores(query, k=5)
    context_text_orgs = "\n\n".join([doc.page_content for doc, _ in results_orgs])

    # Combine contexts
    context_text = context_text_cd + "\n\n" + context_text_grades + "\n\n" + context_text_orgs

    # Format prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # Get response from LLM
    response_text = llm.invoke(prompt).content

    sources = [doc.metadata.get("source", None) for doc, _ in results_cd + results_grades + results_orgs]

    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    return formatted_response, response_text

# Prediction function for Gradio
def predict(message, history):
    history_langchain_format = list()
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    _, response_text = query_rag(message)
    return response_text

# Setting specific size parameters for the Gradio interface
chatbot = gr.ChatInterface(
    fn=predict,
    title="MSU Academic Advisor",
    description="Ask about courses, professors, grades, and student organizations at MSU",
    fill_height=True
)

# Launching the interface with share=True to create a public link
chatbot.launch(share=True)
