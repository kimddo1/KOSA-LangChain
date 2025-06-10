# %%
pip install docx2txt

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import os
from glob import glob

from pprint import pprint
import json

# %%
import docx2txt
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader('./tax.docx')
doc = loader.load()
doc

# %%
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 원본 텍스트 불러오기 (loader.load()는 Document 리스트 반환)
docs = loader.load()  # List[Document]
full_text = "\n".join([doc.page_content for doc in docs])

# 2. 조문 단위로 분할 (조문 제목 포함)
articles = re.split(r'(?=제\d+조(?:의\d+)?\([^)]+\))', full_text)
articles = [a.strip() for a in articles if a.strip()]

# 3. 조문별로 Document 생성
article_docs = [Document(page_content=article) for article in articles]

# 4. RecursiveCharacterTextSplitter 설정
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)

# 5. 각 조문에 대해 청킹
chunked_docs = splitter.split_documents(article_docs)




# %%
# 보고 싶은 청크 번호
chunk_index = 1

print(f'--- 청크 {chunk_index + 1} 전체 내용 (길이: {len(chunked_docs[chunk_index].page_content)}) ---\n')
print(chunked_docs[chunk_index].page_content)
print('--------------------------------------------')


# %%
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings_huggingface = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

chroma_db = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings_huggingface,
    collection_name="tax_test",
    persist_directory="./chroma_db",
    collection_metadata={'hnsw:space':'cosine'}, #l2, ip. cosine
)



# %%
print("총 저장된 문서 수:", chroma_db._collection.count())

# %%
retriever = chroma_db.as_retriever(search_kwargs={"k": 3})


# %%
# def format_docs(docs):
#     titles = []
#     for doc in docs:
#         text = doc.page_content.strip()
#         title = text.split('\n', 1)[0].strip()
#         titles.append(title)
#     unique_titles = list(dict.fromkeys(titles))
#     return "관련 법 조항:\n" + "\n".join(unique_titles)

# retriever_chain = retriever | format_docs
# response = retriever_chain.invoke("납세의무에 대해 알려줘")
# print(response)


# %%
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# %%
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. 프롬프트 템플릿 정의
prompt = PromptTemplate(
template = """
You are a tax expert chatbot. Answer the question based only on the following context.
Do not use any external knowledge.
If the answer is not in the context, say "I don't know".

[Context]
{context}

[Question]
{question}

[Answer]
""",
input_variables=["context", "question"]
)

# 2. RetrievalQA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,                             
    retriever=retriever,                 
    chain_type="stuff",                  
    chain_type_kwargs={"prompt": prompt}
)


# %%
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

def tax_chatbot(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
        
    history_langchain_format.append(HumanMessage(content=message))
    
    response = qa_chain.run(message)
    return response

# 5. Gradio ChatInterface 구성 및 실행
gr.ChatInterface(
    fn=tax_chatbot,
    title="세금 전문 챗봇",
    description="세금 관련 법령에 근거하여 질문에 답변해 드립니다.",
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="세금 관련 질문을 입력하세요.", scale=7),
    submit_btn="질문하기",
).launch()

# %%



