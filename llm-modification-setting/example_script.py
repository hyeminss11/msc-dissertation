import ollama

# 처리할 뉴스 기사 (예시)
news_article = """
[뉴스 기사 원문]
오늘 서울의 날씨는 매우 맑았으며, 낮 최고 기온은 25도에 달했습니다. 
시민들은 공원으로 나와 화창한 날씨를 즐겼습니다.
"""

# 만들고 싶은 프롬프트
# 여기서는 'impersonation'을 테스트 해보겠습니다.
prompt = f"""
다음 뉴스 기사를 '어린 아이'가 설명하는 것처럼 다시 작성해줘. 
친근하고 쉬운 단어를 사용해줘.

뉴스 기사:
{news_article}
"""

try:
    # Ollama 클라이언트에 연결하고, 모델과 프롬프트를 지정하여 요청을 보냅니다.
    response = ollama.chat(
        model='llama3.1:8b',  # Ollama에서 실행한 모델 이름과 동일해야 합니다.
        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ]
    )

    # 응답에서 실제 텍스트 내용만 추출하여 출력합니다.
    generated_text = response['message']['content']
    print("----- 모델이 생성한 텍스트 -----")
    print(generated_text)
    print("-----------------------------")

except Exception as e:
    print(f"오류가 발생했습니다: {e}")