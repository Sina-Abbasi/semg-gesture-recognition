import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

class ProjectRAGEngine:
    def __init__(self, docs_path: str = "."):
        self.docs_path = docs_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def build_index(self):
        documents = []
        target_files = ["README.md"]
        
        reports_dir = os.path.join(self.docs_path, "reports")
        if os.path.exists(reports_dir):
            for file in os.listdir(reports_dir):
                if file.endswith(".txt") or file.endswith(".md"):
                    target_files.append(os.path.join("reports", file))

        for file_path in target_files:
            full_path = os.path.join(self.docs_path, file_path)
            if os.path.exists(full_path):
                loader = TextLoader(full_path, encoding="utf-8")
                documents.extend(loader.load())

        if not documents:
            raise FileNotFoundError("No documentation files found for RAG engine.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )
        chunks = text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

    def ask(self, user_question: str) -> str:
        if not self.vector_store:
            raise ValueError("Vector store is empty. Call build_index() first.")

        # 1. Retrieve relevant context from documentation
        results = self.vector_store.similarity_search(user_question, k=2)
        context = "\n---\n".join([doc.page_content for doc in results])

        # 2. Generate concise answer using Llama 3 via Groq
        prompt = f"""
You are an expert AI assistant for a robotics and sEMG gesture recognition project.
Answer the user's question accurately based ONLY on the provided context below.
Keep your response concise, clear, and limited to 1 or 2 sentences maximum.

Context from documentation:
{context}

Question: {user_question}
Answer:
"""

        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

if __name__ == "__main__":
    rag = ProjectRAGEngine(docs_path=".")
    rag.build_index()
    
    question = "How is RMS calculated and what features are extracted?"
    print(f"\n Question: {question}\n")
    
    answer = rag.ask(question)
    print(" Smart LLM Answer:\n")
    print(answer)