# !wget  https://raw.githubusercontent.com/Unstructured-IO/unstructured/main/example-docs/layout-parser-paper.pdf -P "../../"
#
#
from configs.model_config import *

import nltk

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

from langchain.document_loaders import UnstructuredFileLoader

# loader = UnstructuredFileLoader("./layout-parser-paper.pdf", mode="elements")

##########################################################################
# loader = UnstructuredFileLoader("../content/mg7-demo/MGone.pdf", mode="elements", strategy='fast')
#
# docs = loader.load()
#
# docs[:5]
##########################################################################
# from langchain.document_loaders import PyPDFLoader
#
# loader = PyPDFLoader("../content/mg7-demo/MGone.pdf")
#
# pages = loader.load_and_split()
#
# pages[:5]


##########################################################################
from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./MGone_OCR_TXT.pdf")

data = loader.load()

data[0]
##########################################################################
# from langchain.document_loaders import PDFMinerPDFasHTMLLoader
#
# loader = PDFMinerPDFasHTMLLoader("../content/mg7-demo/MGone.pdf")
#
# data = loader.load()[0]  # entire pdf is loaded as a single Document
#
# from bs4 import BeautifulSoup
#
# soup = BeautifulSoup(data.page_content, 'html.parser')
#
# content = soup.find_all('div')
#
# import re
#
# cur_fs = None
#
# cur_text = ''
#
# snippets = []  # first collect all snippets that have the same font size
#
# for c in content:
#
#     sp = c.find('span')
#
#     if not sp:
#         continue
#
#     st = sp.get('style')
#
#     if not st:
#         continue
#
#     fs = re.findall('font-size:(\d+)px', st)
#
#     if not fs:
#         continue
#
#     fs = int(fs[0])
#
#     if not cur_fs:
#         cur_fs = fs
#
#     if fs == cur_fs:
#
#         cur_text += c.text
#
#     else:
#
#         snippets.append((cur_text, cur_fs))
#
#         cur_fs = fs
#         cur_text = c.text
#
# snippets.append((cur_text, cur_fs))
#
# # Note: The above logic is very straightforward. One can also add more strategies such as removing duplicate snippets (as
# # headers/footers in a PDF appear on multiple pages so if we find duplicatess safe to assume that it is redundant info)
#
# from langchain.docstore.document import Document
#
# cur_idx = -1
# semantic_snippets = []
# # Assumption: headings have higher font size than their respective content
# for s in snippets:
#     # if current snippet's font size > previous section's heading => it is a new heading
#     if not semantic_snippets or s[1] > semantic_snippets[cur_idx].metadata['heading_font']:
#         metadata = {'heading': s[0], 'content_font': 0, 'heading_font': s[1]}
#         metadata.update(data.metadata)
#         semantic_snippets.append(Document(page_content='', metadata=metadata))
#         cur_idx += 1
#         continue
#
#     # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
#     # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
#     if not semantic_snippets[cur_idx].metadata['content_font'] or s[1] <= semantic_snippets[cur_idx].metadata[
#         'content_font']:
#         semantic_snippets[cur_idx].page_content += s[0]
#         semantic_snippets[cur_idx].metadata['content_font'] = max(s[1],
#                                                                   semantic_snippets[cur_idx].metadata['content_font'])
#         continue
#
#     # if current snippet's font size > previous section's content but less tha previous section's heading than also make a new
#     # section (e.g. title of a pdf will have the highest font size but we don't want it to subsume all sections)
#     metadata = {'heading': s[0], 'content_font': 0, 'heading_font': s[1]}
#     metadata.update(data.metadata)
#     semantic_snippets.append(Document(page_content='', metadata=metadata))
#     cur_idx += 1
#
# semantic_snippets[4]
##########################################################################
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="GanymedeNil/text2vec-large-chinese",
    model_kwargs={'device': 'cuda'}
)

db = FAISS.load_local("../vector_store/mg7-demo", embeddings)

query = "行驶中能否调节转向管柱呢"

docs = db.similarity_search(query)

print(docs)
