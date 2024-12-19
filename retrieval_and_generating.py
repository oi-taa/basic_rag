from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain import hub
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from pymilvus import connections
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from pymilvus import Collection

llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    api_key="your_api_key",
    convert_system_message_to_human=True
)
# Connect to Milvus database
connections.connect("default", host="localhost", port="19530")

# Define your prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't get context say you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Define State TypedDict for the workflow
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

#Retrieve via similarity search
def retrieve(state: State):
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    query_embedding = model.encode(state["question"])

    collection = Collection("pdf_collection") 
    collection.load()
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 200}      #effort
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["text"]
    )
    retrieved_docs = []
    for result in results[0]:  
        doc = Document(page_content=result.entity.get("text"))  
        retrieved_docs.append(doc)
    return {"context": retrieved_docs}

# Define the generation function
def generate(state: State):
    docs_content = " ".join(doc.page_content for doc in state["context"])
    messages = custom_rag_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Construct the StateGraph with the retrieve and generate functions
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "Give summary of all info you recieve about Balance Sheet equation?"})

context_text = "\n".join([doc.page_content.strip() for doc in result["context"]])  
print(f'Context:\n{context_text}\n')
print(f'Answer: {result["answer"]}')

