import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from groq import Groq


class ProjectRAGEngine:
    def __init__(self, docs_path: str = "."):
        self.docs_path = docs_path
        self.retriever = None
        
        # Initialize Groq Client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=api_key)

    def load_documents(self):
        docs = []
        # Load README.md 
        readme_path = os.path.join(self.docs_path, "README.md")
        if os.path.exists(readme_path):
            loader = TextLoader(readme_path, encoding="utf-8")
            docs.extend(loader.load())

        # Load any .md or .txt files in reports folder
        reports_dir = os.path.join(self.docs_path, "reports")
        if os.path.exists(reports_dir):
            for file in os.listdir(reports_dir):
                if file.endswith(".md") or file.endswith(".txt"):
                    file_path = os.path.join(reports_dir, file)
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load())
        return docs

    def build_index(self):
        docs = self.load_documents()
        if not docs:
            print("No documentation files found to index.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        # Ultra-lightweight BM25 Search Engine (No PyTorch/SentenceTransformers needed!)
        self.retriever = BM25Retriever.from_documents(chunks)
        self.retriever.k = 3
        print(f"RAG Index built successfully with {len(chunks)} chunks using BM25!")

    def ask(self, query: str) -> str:
        if not self.retriever:
            return "RAG Index is not initialized or no documents were found."

        retrieved_docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""You are an AI assistant helping with a software project.
Answer the following question based ONLY on the provided context.

Context:
{context}

Question: {query}
Answer:"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content