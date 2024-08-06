from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
#from langchain_huggingface import HuggingFaceEmbeddings
#from sentence_transformers import SentenceTransformer
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from re import Pattern
from shutil import rmtree
import os


class RAG:

    def __init__(self, data_path: str = 'data', file_exts: Pattern | str = None, db_path: str = 'chroma', embeddings: str = 'OpenAIEmbeddings'):
        load_dotenv()
        self.data_path = self.set_data_path(data_path)
        self.file_exts = self.set_file_exts(file_exts)
        self.db_path = self.set_db_path(db_path)
        print('Setting embeddings...')
        self.embeddings = self.set_embeddings(embeddings)
        print('loading documents...')
        self.documents = self.load_documents()
        print('splitting documents...')
        self.chunks = self.split_documents()
        print('creating database...')
        self.db = self.create_db()

    def create_db(self):
        db = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
        )
        db.persist()
        return db

    def split_documents(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 250,
            length_function = len,
            add_start_index = True,
        )
        chunks = splitter.split_documents(self.documents)
        return chunks

    def load_documents(self):
        loader = DirectoryLoader(self.data_path, self.file_exts)
        documents = loader.load()
        return documents

    def set_data_path(self, data_path: str):
        if not data_path:
            data_path = 'data'
        if os.path.isfile(data_path):
            self.file_exts = os.path.basename(data_path)
            return os.path.dirname(data_path)
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        if os.path.isdir(data_path):
            return data_path
        raise ValueError(f'{data_path} is not a valid path or does not have read access.')

    def set_file_exts(self, file_exts: Pattern | str) -> str:
        if file_exts is None:
            return "*"
        if isinstance(file_exts, Pattern):
            return f"{file_exts}"
        if not isinstance(file_exts, str):
            raise ValueError(f'file_exts must be a string or a compiled regular expression.')
        return file_exts
    
    def set_db_path(self, db_path: str):
        if os.path.isdir(db_path):
            rmtree(db_path)
        os.mkdir(db_path)

        return db_path

    def set_embeddings(self, embedding: str):
        if embedding == 'SentenceTransformerEmbeddings':
            pass # TODO: Troubleshoot SentenceTransformerEmbeddings import errors
            #return SentenceTransformer()
        if embedding == 'OpenAIEmbeddings':
            return OpenAIEmbeddings()
        raise ValueError(f'Embedding {embedding} not supported.')

if __name__ == '__main__':
    rag = RAG()
    print(rag.data_path)
    print(rag.file_exts)
    print(rag.db_path)
    print(rag.embeddings)