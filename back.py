from dotenv import load_dotenv
import os
from FlagEmbedding import FlagReranker
import heapq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.runnables import RunnableLambda


class LLMHandler:
    def __init__(self, model_name: str, gemini_key: str):
        self.api_key = gemini_key
        self.llm = ChatGoogleGenerativeAI(model=model_name, api_key=self.api_key)
    
    def get_llm(self):
        return self.llm
class VectorDatabase:
    def __init__(self, model_name: str, collection_name: str, db_path: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.collection_name = collection_name
        self.db_path = db_path
        self.db = self.load_db()
        
    def load_db(self):
        return QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            path=self.db_path
        )
    def get_retriever(self):
        return self.db


class QuestionAnsweringChain:
    def __init__(self, llm_handler: LLMHandler, vector_db: VectorDatabase, num_docs: int = 5, apply_rewrite: bool = False, apply_rerank: bool = False):
        self.num_docs = num_docs
        self.llm = llm_handler.get_llm()
        self.db = vector_db.get_retriever()
        self.memory = []
        if apply_rerank:
            self.retriever = self.db.as_retriever(search_kwargs={"k": num_docs * 2})
        else:
            self.retriever = self.db.as_retriever(search_kwargs={"k": num_docs})
        self.output_parser = StrOutputParser()
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            Bạn là chatbot thông minh. Dựa vào những thông tin dưới đây, nếu không có dữ liệu liên quan đến câu hỏi, hãy trả lời 'Chúng tôi không có thông tin', ngoài ra có thể có 1 số câu hỏi không cần thôn tin dưới, hãy trả lời tự nhiên:
            {context}

            Lịch sử hội thoại:
            {chat_history}

            Hãy trả lời câu hỏi sau: {question}
            """
        )
        self.reranker = FlagReranker('namdp-ptit/ViRanker', use_fp16=True)

        self.chain = self.create_chain(apply_rewrite=apply_rewrite, apply_rerank=apply_rerank)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def ReWrite(self, query):
        template = f'''
        Viết lại câu hỏi dưới đây sao cho rõ ràng, chính xác và phù hợp với ngữ cảnh tìm kiếm. Thêm vào đó, cung cấp một chút gợi ý và suy luận để tăng khả năng tìm kiếm và trả lời từ cơ sở dữ liệu. Đảm bảo câu hỏi mới vẫn giữ nguyên ý nghĩa của câu hỏi gốc.

        Câu hỏi gốc: "{query}"

        Câu hỏi viết lại:
        '''
        rewrite_query = self.llm.invoke(template)
        return rewrite_query.content

    def ReRank(self, query_docs):
        query = query_docs['query']
        chunks = query_docs['docs']
        top_k = self.num_docs
        scores = self.reranker.compute_score(
            [[query, chunk.page_content] for chunk in chunks],
            normalize=True
        )
        chunk_with_rank = [(chunks[idx], scores[idx]) for idx in range(len(chunks))]
        top_chunks = heapq.nlargest(top_k, chunk_with_rank, key=lambda x: x[1])
        return [chunk for chunk, score in top_chunks]

    def find_neighbor(self, docs):
        for doc in docs:
            doc_id = doc.metadata['_id']
            neighbor_ids = [doc_id - 2, doc_id - 1, doc_id + 1, doc_id + 2]
            neighbors = self.db.get_by_ids(neighbor_ids)
            neighbors.append(doc)
            neighbors_sorted = sorted(neighbors, key=lambda x: x.metadata['_id'])
            doc.page_content = '\n'.join([neighbor.page_content for neighbor in neighbors_sorted])
        return docs

    def get_chat_history(self):
        return '\n'.join(self.memory) if self.memory else ""


    def create_chain(self, apply_rewrite: bool = False, apply_rerank: bool = False):
        retriever_handler = self.retriever
        if apply_rewrite:
            pre_retriever = self.ReWrite
        else:
            pre_retriever = RunnablePassthrough()
        if apply_rerank:
            retriever_handler = RunnableParallel(
                {'docs': retriever_handler, 'query': RunnablePassthrough()}
            )
            retriever_handler = retriever_handler | self.ReRank
        retriever_handler = retriever_handler | self.find_neighbor | self.format_docs
        chat_history_handler = RunnableLambda(lambda x: self.get_chat_history())
        setup_and_retrieval = RunnableParallel(
            {"context": retriever_handler, "question": RunnablePassthrough(), 'chat_history': chat_history_handler}
        )
        chain = pre_retriever | setup_and_retrieval | self.prompt_template | self.llm | self.output_parser
        return chain

    def run(self, question: str):
        # Lưu lịch sử hội thoại
        self.memory.append(f'người dùng: {question}')
        response = self.chain.invoke(question)

        # Cập nhật phản hồi vào lịch sử hội thoại
        self.memory.append(f'chatbot: {response}')
        if len(self.memory) > 3:
            self.memory.pop(0)
            self.memory.pop(0)
        return response
