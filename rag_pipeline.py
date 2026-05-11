import os
import time
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()


VECTOR_DB_PATH = "vectorstore"


def load_pipeline():
    # Embeddings
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS index
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        hf_embeddings,
        allow_dangerous_deserialization=True
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        max_tokens=2048
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""
You are a medical consultant with expertise in understanding doctor-patient conversations,
symptom descriptions, and medical chat transcripts.

Use ONLY the information provided from the "Medical Care and Chats" dataset.
Stay strictly within the given chats, symptoms, diagnoses, and conversation notes.

If relevant information exists, provide a clear, short, medically accurate response.
If the answer is not present, respond exactly with:
"The answer is not available in provided context."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""
    )

    # Conversational RAG chain (classic)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )

    return chain


def ask_question(chain, question: str) -> dict:
    start_time = time.time()

    result = chain.invoke({
        "question": question,
        "chat_history": []
    })

    latency = time.time() - start_time

    docs = result["source_documents"]
    retrieved_docs = [doc.page_content[:200] for doc in docs]
    sources = [doc.metadata for doc in docs]

    return {
        "answer": result["answer"],
        "retrieved_docs": retrieved_docs,
        "sources": sources,
        "latency": latency
    }


if __name__ == "__main__":
    chain = load_pipeline()
    response = ask_question(
        chain,
        "What symptoms and diagnoses are mentioned for diabetes in the medical chats?"
    )

    print(response["answer"])
    print(response["sources"])
    print("Latency:", response["latency"])