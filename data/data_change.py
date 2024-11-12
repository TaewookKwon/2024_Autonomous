import pandas as pd
index = 6
# 파일 경로 설정
file_path = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/evaluation/future_trajectory_{index}.csv'

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 삭제할 열의 인덱스 계산
delete_col1 = 1 + 2 * 31 * 4
delete_col2 = delete_col1 + 1

# 열 삭제
df.drop(df.columns[[delete_col1, delete_col2]], axis=1, inplace=True)

# 수정된 파일 저장
df.to_csv(f'/home/taewook/catkin_ws/src/beginner_tutorials/data/evaluation/future_trajectory_{index}.csv', index=False)