from langchain_community.vectorstores import Chroma              
from langchain_community.document_loaders import PyPDFLoader     
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 

def embed_and_store(pdf_path, persist_directory):
    """
    Convert text to vector
    
    Parameters:
    -----------
    pdf_path : pdf
        Location file pdf will convert
    
    persist_directory : url
        Url folder when chroma done create vector
    """
    # load file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # chunky all document
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # using sentence transform
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # create embedding
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # save memory data
    vectordb.persist()

    print(f"✅ Disimpan ke ChromaDB di: {persist_directory}")

# run
embed_and_store(
    pdf_path="LapKeu 0924.pdf", 
    persist_directory="C:/Project/Project Pribadi/chroma_db" 
)
