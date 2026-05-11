import os
import time
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain   # ✅ FIXED

load_dotenv()

VECTORSTORE_PATH = "vectorstore"


def load_pipeline():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load vector DB
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # LLM
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
You are a medical consultant.
Answer ONLY from the provided context.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""
    )

    # Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )

    return chain


def ask_question(chain, question: str) -> dict:
    start = time.time()

    result = chain.invoke({
        "question": question,
        "chat_history": []
    })

    latency = time.time() - start

    docs = result["source_documents"]

    return {
        "answer": result["answer"],
        "sources": [doc.metadata for doc in docs],
        "latency": latency
    }
