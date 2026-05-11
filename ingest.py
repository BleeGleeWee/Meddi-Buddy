from langchain.document_loaders import JSONLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()


def build_index():
    jq_schema = ".[] | {instruction: .instruction, input: .input, output: .output}"

    loader = JSONLoader(
        file_path="./data/chatdoctor5k.json",
        jq_schema=jq_schema,
        text_content=False
    )
    medical_json_docs = loader.load()

    csv_loader = CSVLoader(file_path="./data/format_dataset.csv")
    medical_csv_docs = csv_loader.load()

    medical_docs = medical_csv_docs + medical_json_docs

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = splitter.split_documents(medical_docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)

    # ✅ IMPORTANT CHANGE
    vectorstore.save_local("vectorstore")

    print("✅ Vectorstore created successfully!")


if __name__ == "__main__":
    build_index()
