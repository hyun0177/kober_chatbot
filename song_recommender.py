import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from anal import anal_lyric

def recommend_song(user_emotion_class, user_emotion_vector):
    # CSV 파일 읽기
    csv_file_path = 'lyricanal.csv'  # 실제 CSV 파일 경로
    song_data = pd.read_csv(csv_file_path)

    # 감정 클래스 필터링
    filtered_songs = song_data[song_data['emotion'] == user_emotion_class]

    # 유사도 계산을 위한 리스트 초기화
    max_similarity = -1  # 최댓값 초기화
    recommended_title = None
    recommended_artist = None

    user_emotion_vector = np.array(user_emotion_vector)

    for _, song in filtered_songs.iterrows():
        # CSV의 벡터값을 문자열에서 NumPy 배열로 변환
        song_emotion_vector = np.array(eval(song['emotion_vector']))  # 문자열을 리스트로 변환

        similarity = cosine_similarity([user_emotion_vector], [song_emotion_vector])[0][0]

        # 최대 유사도 업데이트
        if similarity > max_similarity:
            max_similarity = similarity
            recommended_title = song['title']
            recommended_artist = song['singer']

    
    return recommended_title, recommended_artist

def main():
    # 사용자 입력
    user_emotion_class, user_emotion_vector = anal_lyric()
    recommended_title, recommended_artist = recommend_song(user_emotion_class, user_emotion_vector)

    if recommended_title and recommended_artist:
        print(f"추천하는 노래: {recommended_title} - {recommended_artist}")
    else:
        print("추천할 노래가 없습니다.")

if __name__ == "__main__":
    main()