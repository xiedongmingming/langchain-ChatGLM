import logging
import sys
import json

from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# embeddings = HuggingFaceEmbeddings(  # 'GanymedeNil/text2vec-large-chinese'
#     model_name='F:\\cache\\torch\\sentence_transformers\\GanymedeNil_text2vec-large-chinese',
#     model_kwargs={'device': 'cuda'}
# )


embeddings = HuggingFaceEmbeddings(
    model_name="rainjay/sbert_nlp_corom_sentence-embedding_chinese-base-ecom",
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

    document = data['document'][0]

    for pdfelements in data['pdf_elements']:

        if 'elements' in pdfelements:

            for elements in pdfelements['elements']:

                if 'element_type' in elements and 'text' in elements and 'page' in elements:

                    if elements['element_type'] == 'paragraphs':
                        #
                        # print('===========================》》》》》》》》》》》')
                        # print(elements['text'])

                        # 为每个段落生成总结

                        documents.append(
                            Document(
                                page_content=elements['text'],
                                metadata=dict(
                                    page=elements['page'],
                                    syllabus=elements['syllabus'],
                                    source=document['filename']
                                )
                            )
                        )

                else:

                    if elements['element_type'] not in ['images', 'tables']:
                        #
                        print('——————————————————————> ', elements['element_type'])

newdocs = []

syllabus = None

for doc in documents:

    if syllabus and syllabus == doc.metadata['syllabus']:

        newdocs[-1].page_content += '\n{}'.format(doc.page_content)

    else:

        newdocs.append(Document(
            page_content=doc.page_content,
            metadata=dict(
                page=doc.metadata['page'],
                syllabus=doc.metadata['syllabus'],
                source=doc.metadata['source']
            )
        ))

    syllabus = doc.metadata['syllabus']

db = FAISS.from_documents(newdocs, embeddings)

db.save_local('./vector_store2')
