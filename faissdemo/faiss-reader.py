from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from typing import List, Tuple

import numpy as np

from utils import torch_gc

from langchain.docstore.document import Document


def load_vector_store(vs_path, embeddings):
    #
    return FAISS.load_local(vs_path, embeddings)


def seperate_list(ls: List[int]) -> List[List[int]]:
    #
    lists = []

    ls1 = [ls[0]]

    for i in range(1, len(ls)):
        #
        if ls[i - 1] + 1 == ls[i]:

            ls1.append(ls[i])

        else:

            lists.append(ls1)

            ls1 = [ls[i]]

    lists.append(ls1)

    return lists


def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
) -> List[Tuple[Document, float]]:
    #
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)

    # 查询结果：
    # scores: ndarray(1, 4)
    # indices: ndarray(1, 4)

    docs = []  # 最终结果

    id_set = set()

    store_len = len(self.index_to_docstore_id)

    for j, i in enumerate(indices[0]):

        if i == -1 or 0 < self.score_threshold < scores[0][j]:  # This happens when not enough docs are returned.
            #
            continue

        _id = self.index_to_docstore_id[i]  # docstore_id

        doc = self.docstore.search(_id)  # Document(page_content='表 2：仰望 U8 主要竞品对比', metadata={'page': 22})

        if not self.chunk_conent:

            if not isinstance(doc, Document):
                #
                raise ValueError(f"Could not find document for id {_id}, got {doc}")

            doc.metadata["score"] = int(scores[0][j])

            docs.append(doc)

            continue

        id_set.add(i)

        docs_len = len(doc.page_content)  # 原始文档长度

        for k in range(1, max(i, store_len - i)):  # 前后补充

            break_flag = False

            for l in [i + k, i - k]:  # 左右各一个尝试

                if 0 <= l < len(self.index_to_docstore_id):

                    _id0 = self.index_to_docstore_id[l]

                    doc0 = self.docstore.search(_id0)

                    if docs_len + len(doc0.page_content) > self.chunk_size:

                        break_flag = True

                        break

                    elif doc0.metadata["source"] == doc.metadata["source"]:  # 同一页？？？

                        docs_len += len(doc0.page_content)

                        id_set.add(l)

            if break_flag:
                #
                break

    if not self.chunk_conent:
        #
        return sorted(docs, key=lambda d: 1000 - d.metadata['score'])

    if len(id_set) == 0 and self.score_threshold > 0:
        #
        return []

    id_list = sorted(list(id_set))

    id_lists = seperate_list(id_list)

    for id_seq in id_lists:

        for id in id_seq:

            if id == id_seq[0]:

                _id = self.index_to_docstore_id[id]

                doc = self.docstore.search(_id)

            else:

                _id0 = self.index_to_docstore_id[id]

                doc0 = self.docstore.search(_id0)

                doc.page_content += " " + doc0.page_content

        if not isinstance(doc, Document):
            #
            raise ValueError(f"Could not find document for id {_id}, got {doc}")

        doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])

        doc.metadata["score"] = int(doc_score)

        docs.append(doc)

    torch_gc()

    return sorted(docs, key=lambda d: 1000 - d.metadata['score'])


# embeddings = HuggingFaceEmbeddings(  # 'GanymedeNil/text2vec-large-chinese'
#     model_name='F:\\cache\\torch\\sentence_transformers\\GanymedeNil_text2vec-large-chinese',
#     model_kwargs={'device': 'cuda'}
# )

embeddings = HuggingFaceEmbeddings(
    model_name="rainjay/sbert_nlp_corom_sentence-embedding_chinese-base-ecom",
    model_kwargs={'device': 'cuda'}
)

FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector

# vector_store = load_vector_store('./vector_store1', embeddings)
vector_store = load_vector_store('./vector_store2', embeddings)

vector_store.chunk_conent = False
vector_store.score_threshold = 0
vector_store.chunk_size = 500

related_docs_with_score = vector_store.similarity_search_with_score('仰望U8有哪些主要竞品', k=2)

print(related_docs_with_score)

