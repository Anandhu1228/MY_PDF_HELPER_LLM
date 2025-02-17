from fastapi import FastAPI, HTTPException, Request
import os
from pydantic import BaseModel
import logging
import json

app = FastAPI()

from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.llms import HuggingFaceHub
from AI_GATEWAYS import huggingface_api_key
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings


from helpers import load_chunks, calculate_confidence, rerank_chunks, format_chunks
from extractor import extracter_ttp
embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")


memory = ConversationBufferWindowMemory(
    k=10,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)


llm = HuggingFaceHub(
    repo_id="deepseek-ai/DeepSeek-R1",
    # repo_id="meta-llama/Llama-3.3-70B-Instruct",
    model_kwargs={"temperature": 0.2, "max_length": 1024},
    huggingfacehub_api_token=huggingface_api_key
)


CONTEXT_FILE = "context_topic.json"


chunks = load_chunks()
persist_directory = os.path.join(os.getcwd(), "chroma_db_pedition")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)



def load_context():
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r") as f:
            return json.load(f).get("contextTopic", "default")
    return "default"



def save_context(contextTopic):
    with open(CONTEXT_FILE, "w") as f:
        json.dump({"contextTopic": contextTopic}, f)

contextTopic = load_context()



def update_qa_prompt(new_topic):
    global qa_prompt
    qa_prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Answer the user's question based on the conversation history and the provided {{new_topic}} context. Not from external source. 

        ### Chat History:
        {chat_history}

        ### Context:
        {context}

        ### Question: 
        {question}

        ### Answer:
        """
    )
    save_context(new_topic)



def initialize_qa_chain():
    return (
        {"context": lambda x: x["chunks"],
         "question": lambda x: x["question"],
         "chat_history": lambda x: x["chat_history"]}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

update_qa_prompt(contextTopic)
qa_chain = initialize_qa_chain()



def ask_question(question):
    # Retrieve context
    initial_chunks = ensemble_retriever.get_relevant_documents(question)
    final_chunks, relevance_scores = rerank_chunks(question, initial_chunks, top_k=3)
    
    # Calculate confidence
    confidence = calculate_confidence(relevance_scores)
    
    # Generate answer
    raw_answer = qa_chain.invoke({
        "question": question,
        "chunks": format_chunks(final_chunks),
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })

    # Extract everything after "### Answer:"
    answer = raw_answer.split("### Answer:")[-1].strip()
    
    # Store interaction in memory
    memory.save_context({"question": question}, {"answer": answer})
    
    # Add confidence and sources
    sources = list(set(c.metadata["page"] for c in final_chunks))
    response = f"{answer}\n\nConfidence: {confidence:.0%}\nSources: Pages {', '.join(map(str, sources))}"
    
    return response



# EG FOR AN ODYSSEY RELATED CONTEXT CONCEPT
# print(ask_question("who wrote odyssey?"))



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class QuestionRequest(BaseModel):
    text: str



@app.post("/ask_question")
async def ask(request: Request):
    try:
        body = await request.body()
        text = body.decode('utf-8')  # Decode the raw bytes to string
        response = ask_question(text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/fit_content")
async def fit_content(request: Request):
    try:
        body = await request.json()
        new_topic = body.get("contextTopic", "").strip()
        pdf_present = body.get("pdfPresent", False) 
        if not new_topic:
            raise HTTPException(status_code=400, detail="Invalid context topic")

        update_qa_prompt(new_topic)

        if pdf_present:
            extracter_ttp()

        return {"message": "Content processed successfully", "contextTopic": new_topic}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


print(load_context())