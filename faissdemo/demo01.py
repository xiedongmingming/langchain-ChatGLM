import os
import getpass

# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader

from configs.model_config import *

import datetime

from textsplitter import ChineseTextSplitter

from typing import List, Tuple

from langchain.docstore.document import Document

import numpy as np

from utils import torch_gc
from tqdm import tqdm
from pydantic import BaseModel, Field
from pypinyin import lazy_pinyin

from langchain.document_loaders import TextLoader

# loader = TextLoader('../../../state_of_the_union.txt')
#
# documents = loader.load()
#
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#
# docs = text_splitter.split_documents(documents)

import nltk

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

loader = UnstructuredFileLoader('./合并测试.pdf', strategy="fast")

textsplitter = ChineseTextSplitter(pdf=True, sentence_size=100)

docs = loader.load_and_split(textsplitter)

embeddings: BaseModel = None

embeddings = HuggingFaceEmbeddings(
    model_name="GanymedeNil/text2vec-large-chinese",
    model_kwargs={'device': 'cuda'}
)

db = FAISS.from_documents(docs, embeddings)

db.save_local('F:\\workspace\\github\\xiedongmingming\\LangChain-ChatGLM\\faissdemo\\vectorstore')

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

print(docs[0].page_content)
