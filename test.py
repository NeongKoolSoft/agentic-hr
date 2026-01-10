import os
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 구글 API 키 설정 (따옴표 안에 키를 넣어주세요)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBNj3DDVhaM5uYETqdJSMAwYX0NjlKp86k" 

# 2. Gemini 모델 준비 (가볍고 빠른 gemini-1.5-flash 모델 사용)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 3. 질문 던지기
print("Gemini에게 질문하는 중...")
try:
    answer = llm.invoke("ERP 개발 경험을 살려 AI 엔지니어가 되려면 뭐부터 해야 할까? 3줄 요약해줘.")
    
    # 4. 대답 출력
    print("-" * 30)
    print(answer.content)
    print("-" * 30)

except Exception as e:
    print("에러 발생:", e)