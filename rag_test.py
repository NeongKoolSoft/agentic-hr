import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

BASE_DIR = Path(__file__).parent
MANUAL_PATH = BASE_DIR / "manual.txt"

loader = TextLoader(MANUAL_PATH, encoding="utf-8")

# 1. API 키 설정 (⚠️ 키는 코드에 박지 말고 환경변수로 권장)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBNj3DDVhaM5uYETqdJSMAwYX0NjlKp86k"

print("1. 문서 읽는 중...")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("2. 벡터 DB에 저장 중...")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template(
    """너는 제공된 컨텍스트만 사용해서 답해.
컨텍스트:
{context}

질문: {question}
답변:"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "넝쿨OS의 음성 호출 명령어는 뭐야?"
print(f"3. 질문: {question}")
print("-" * 30)
result = rag_chain.invoke(question)
print("답변:", result)
