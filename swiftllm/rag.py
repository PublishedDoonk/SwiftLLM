from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import SentenceTransformersEmbeddings
from re import Pattern
import os


class RAG:

    def __init__(self, data_path: str = 'data', file_exts: Pattern | str = None, db_path: str = 'chroma', embeddings: str = 'SentenceTransformersEmbeddings'):
        self.data_path = self.set_data_path(data_path)
        self.file_exts = self.set_file_exts(file_exts)
        self.db_path = self.set_db_path(db_path)
        self.embeddings = self.set_embeddings(embeddings)

    def set_data_path(self, data_path: str):
        if not data_path:
            data_path = 'data'
        if os.path.isdir(data_path):
            return data_path
        if os.path.isfile(data_path):
            self.file_exts = os.path.basename(data_path)
            return os.path.dirname(data_path)
        raise ValueError(f'{data_path} is not a valid path or does not have read access.')

    def set_file_exts(self, file_exts: Pattern | str):
        if file_exts is None:
            return r'.*\.md'
        if isinstance(file_exts, str):
            return r'.*' + file_exts
        if not isinstance(file_exts, Pattern):
            raise ValueError(f'file_exts must be a string or a compiled regular expression.')
        return file_exts
    
    def set_db_path(self, db_path: str):
        if os.path.isdir(db_path):
            os.removedirs(db_path)
        os.mkdir(db_path)

        return db_path

    def set_embeddings(self, embedding: str):
        if embedding == 'SentenceTransformersEmbeddings':
            return SentenceTransformersEmbeddings(model_name='all-MiniLM-L6-v2')
        if embedding == 'OpenAIEmbeddings':
            return OpenAIEmbeddings()
        raise ValueError(f'Embedding {embedding} not supported.')

    