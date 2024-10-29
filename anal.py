# test.py
import torch
from emotion_model.model import load_model, analyze_emotion  # model.py에서 함수 가져오기

# 감정 레이블 설정 (예: 감정의 수에 맞게 조정 필요)
emotion_labels = ['분노', '행복', '불안', '당황', '슬픔', '중립', '혐오']  # 실제 감정 레이블에 맞게 수정하세요

def anal_lyric():
    # 모델과 토크나이저 로드
    model, tokenizer = load_model()
    
    input_sentences = []  # 문장을 저장할 리스트

    while True:
        # 사용자로부터 입력 받기
        input_text = input("감정 분석할 문장을 입력하세요 (종료하려면 'exit' 입력): ")
        if input_text.lower() == 'exit':
            break
        input_sentences.append(input_text)  # 입력받은 문장 추가

    # 모든 문장을 하나의 문자열로 합치기
    combined_text = ' '.join(input_sentences)

    # 감정 분석
    emotion_scores = analyze_emotion(combined_text, model, tokenizer)

    # 가장 높은 감정 점수와 해당 감정 레이블 찾기
    max_score_index = emotion_scores.index(max(emotion_scores))
    closest_emotion = emotion_labels[max_score_index]

    return closest_emotion, emotion_scores