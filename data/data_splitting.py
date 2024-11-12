import pandas as pd
import numpy as np

# training, test 비율
ratio = 0.9
# 파일 경로
processed_file_path = '/home/taewook/catkin_ws/src/beginner_tutorials/data/processed_data/dataset_ego.csv'
processed_data = pd.read_csv(processed_file_path, header=None)

# 데이터 섞기
shuffled_data = processed_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 데이터 분할 비율 설정
train_size = int(ratio * len(shuffled_data))  # 90%는 훈련 데이터
test_size = len(shuffled_data) - train_size  # 10%는 테스트 데이터

# 훈련 데이터와 테스트 데이터 분할
train_data = shuffled_data.iloc[:train_size]
test_data = shuffled_data.iloc[train_size:]

# 파일로 저장
train_data.to_csv('/home/taewook/catkin_ws/src/beginner_tutorials/data/processed_data/training_ego.csv', header=False, index=False)
test_data.to_csv('/home/taewook/catkin_ws/src/beginner_tutorials/data/processed_data/validation_ego.csv', header=False, index=False)

print(f'Training data size: {len(train_data)}, Validation data size: {len(test_data)}')
