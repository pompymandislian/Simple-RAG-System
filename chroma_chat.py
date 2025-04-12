from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from openai import AzureOpenAI
import json

# Inisialisasi memory buffer
memory = ConversationBufferMemory(return_messages=True)

def convert_history_to_dict(chat_history):
    """Konversi HumanMessage dan AIMessage ke format OpenAI API"""
    formatted = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted.append({"role": "assistant", "content": msg.content})
    return formatted

def chat(query, persist_directory="C:/Project/Project Pribadi/chroma_db", k=3):
    """Create Chat"""
    # Load config Azure OpenAI
    with open("config.json", "r") as f:
        config = json.load(f)

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load persisted Chroma
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Retrieve context dari vectordb
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # Ambil riwayat percakapan dari memory
    raw_chat_history = memory.load_memory_variables({})["history"]
    chat_history = convert_history_to_dict(raw_chat_history)

    # Join context
    messages = [
        {"role": "system", "content": f"Gunakan konteks berikut untuk menjawab pertanyaan:\n\n{context}"}
    ] + chat_history + [
        {"role": "user", "content": query}
    ]

    # Azure OpenAI Chat Client
    client = AzureOpenAI(
        api_key=config["api_key"],
        azure_endpoint=config["azure_endpoint"],
        api_version=config["api_version"]
    )

    # Streaming response
    stream = client.chat.completions.create(
        model=config["deployment_name"],
        messages=messages,
        temperature=0.7,
        max_tokens=1500,
        stream=True
    )

    # Gabungkan hasil dan simpan ke memory
    full_reply = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content_piece = chunk.choices[0].delta.content
            full_reply += content_piece
            yield content_piece

    # Simpan percakapan ke memory
    memory.save_context(
        {"input": query},
        {"output": full_reply}
    )

# # Run
# if __name__ == "__main__":
#     jawaban = simple_rag_with_chroma("30 September PT ramayana seperti apa ya?")
#     print(jawaban)
