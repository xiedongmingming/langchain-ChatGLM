import logging
import sys
import json

from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

embeddings = HuggingFaceEmbeddings(  # 'GanymedeNil/text2vec-large-chinese'
    model_name='F:\\cache\\torch\\sentence_transformers\\GanymedeNil_text2vec-large-chinese',
    model_kwargs={'device': 'cuda'}
)


#########################################################################################
# documents = SimpleDirectoryReader(
#     input_files=['F:\\workspace\\github\\adwetec.com\\chatglm\\graphdemo\\汽车行业上海车展新车专题.pdf']
# ).load_data()
def read_file(file_path):
    #
    with open(file_path, 'rb') as file:
        #
        return file.read()


contents = read_file("./识别结果.txt")

data = json.loads(contents)

documents = []

if 'pdf_elements' in data:

    for pdfelements in data['pdf_elements']:

        if 'elements' in pdfelements:

            for elements in pdfelements['elements']:

                if 'element_type' in elements and 'text' in elements and 'page' in elements:

                    if elements['element_type'] == 'paragraphs':
                        #
                        print('===========================》》》》》》》》》》》')
                        print(elements['text'])

                        documents.append(
                            Document(page_content=elements['text'], metadata=dict(page=elements['page']))
                        )

                else:

                    print(elements)

db = FAISS.from_documents(documents, embeddings)

db.save_local('./vector_store')
