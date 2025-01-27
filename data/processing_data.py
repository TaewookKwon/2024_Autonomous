import pandas as pd
import numpy as np

# 값 초기화
index_start = 1 #1
index_finish = 30  # 30 # 몇번째까지 프로세싱
dt = 0.1  # [s] 예측할 time step
prediction_time = 3  # [s]
look_back = int(prediction_time / dt)
print(look_back)
look_ahead = look_back
skip_rows = 3  # 0.1 * 3 0.3초씩 skip

def process_ego_and_lane_data(ego_file, skip_rows, look_back, look_ahead):
    # 데이터 로드 (헤더 없음)
    ego_data = pd.read_csv(ego_file, header=None)

    result = []
    
    # ego_tracking_data.csv 파일의 줄마다 처리
    for i in range(look_back, len(ego_data) - look_ahead, skip_rows):
        current_row = ego_data.iloc[i]
        current_time = current_row[0]  # ego_data의 첫 번째 열이 'time'

        # 현재 줄을 포함한 30개 전 데이터 가져오기 (현재 줄 포함)
        if i - look_back >= 0:
            past_data = ego_data.iloc[i-look_back+1:i+1, 1:].values.flatten()  # [x, y, velocity, yaw, x1, x2, x3] 사용
        else:
            continue  # 30번째 전 데이터가 없는 경우 스킵

        # 30개 후 데이터 가져오기
        if i + look_ahead < len(ego_data):
            future_data = ego_data.iloc[i+1:i+1+look_ahead, [1, 2, 4]].values.flatten()  # [x, y, yaw] 사용
        else:
            continue  # 30번째 후 데이터가 없는 경우 스킵

        # 결과 저장: past_data -> future_data 순으로 저장
        result_row = np.concatenate([past_data, future_data])
        result.append(result_row)

    return pd.DataFrame(result)

# 전체 데이터프레임을 저장할 리스트
all_processed_data = []

for i in range(index_start, index_finish + 1):
    # 파일 경로 설정
    ego_file = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/tracking/ego_tracking_data_{i}.csv'

    # 데이터 처리
    processed_data = process_ego_and_lane_data(ego_file, skip_rows, look_back, look_ahead)

    # 처리한 데이터를 전체 데이터프레임에 추가
    all_processed_data.append(processed_data)

# 모든 데이터프레임을 하나로 합침
final_dataset = pd.concat(all_processed_data, ignore_index=True)

# 결과를 CSV 파일로 저장
final_dataset.to_csv('/home/taewook/catkin_ws/src/beginner_tutorials/data/processed_data/validation1.csv', index=False, header=False)



# import pandas as pd
# import numpy as np

# # def find_nearest_lane_time(lane_data, ego_time):
# #     # ego_time과 가장 가까운 lane_time을 찾기 위한 함수
# #     min_diff = np.inf
# #     best_row = None
# #     for i in range(len(lane_data)):
# #         time_diff = abs(lane_data.iloc[i, 0] - ego_time)  # lane_data의 첫 번째 열이 'time'
# #         if time_diff < min_diff:
# #             min_diff = time_diff
# #             best_row = i
# #         else:
# #             break  # 차이가 커지면 멈춘다 (정렬되어 있다고 가정)
# #     return best_row

# # 값 초기화

# index_start = 6
# index_finish = 6 # 몇번째까지 프로세싱

# dt = 0.1 # [s] 예측할 time step
# prediction_time = 3 # [s]
# look_back = int(prediction_time / dt)
# print(look_back)
# look_ahead = look_back
# skip_rows=3 # 0.2 * 3 0.6초씩 skip

# def process_ego_and_lane_data(ego_file, skip_rows, look_back, look_ahead):
#     # 데이터 로드 (헤더 없음)
#     ego_data = pd.read_csv(ego_file, header=None)

#     result = []
    
#     # ego_tracking_data.csv 파일의 줄마다 처리
#     for i in range(look_back, len(ego_data) - look_ahead, skip_rows):
#         current_row = ego_data.iloc[i]
#         current_time = current_row[0]  # ego_data의 첫 번째 열이 'time'

#         # 현재 줄을 포함한 30개 전 데이터 가져오기 (현재 줄 포함)
#         if i - look_back >= 0:
#             past_data = ego_data.iloc[i-look_back+1:i+1, 1:].values.flatten()  # [x, y, velocity, yaw, x1, x2, x3] 사용
#         else:
#             continue  # 30번째 전 데이터가 없는 경우 스킵

#         # 30개 후 데이터 가져오기
#         if i + look_ahead < len(ego_data):
#             future_data = ego_data.iloc[i+1:i+1+look_ahead, [1,2,4]].values.flatten()  # [x, y, yaw] 사용
#         else:
#             continue  # 30번째 후 데이터가 없는 경우 스킵

#         # 결과 저장: past_data -> future_data -> lane_info 순으로 저장
#         result_row = np.concatenate([past_data, future_data])
#         result.append(result_row)

#     return pd.DataFrame(result)

# for i in range(index_start, index_finish + 1):
#     # 파일 경로 설정
#     ego_file = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/tracking/ego_tracking_data_{i}.csv'

#     # 데이터 처리
#     processed_data = process_ego_and_lane_data(ego_file, skip_rows, look_back, look_ahead)

#     # 결과를 CSV 파일로 저장
#     processed_data.to_csv(f'/home/taewook/catkin_ws/src/beginner_tutorials/data/trajectory_data_processed_{i}.csv', index=False, header=False)
