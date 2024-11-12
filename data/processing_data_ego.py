import pandas as pd
import numpy as np

# 값 초기화
index_start = 1 #1
index_finish = 62  # 30 # 몇번째까지 프로세싱
dt = 0.1  # [s] 예측할 time step
prediction_time = 3  # [s]
look_back = int(prediction_time / dt)
#print(look_back)
look_ahead = look_back
skip_rows = 3  # 0.1 * 3 0.3초씩 skip

def process_ego_and_lane_data(ego_file, skip_rows, look_back, look_ahead):
    # 데이터 로드 (헤더 없음)
    ego_data = pd.read_csv(ego_file, header=None)

    result = []
    
    # ego_tracking_data.csv 파일의 줄마다 처리
    for i in range(look_back, len(ego_data) - look_ahead, skip_rows):
        current_row = ego_data.iloc[i]

        # # 현재 데이터로부터 offset 생성 (30, 7) 형태로
        # offset_value = np.array([ego_data.iloc[i, 1], ego_data.iloc[i, 2], 0, 0, ego_data.iloc[i, 1], ego_data.iloc[i, 1], ego_data.iloc[i, 1]])
        # offset = np.tile(offset_value, (look_back, 1))  # (30, 7) 형태로 복제

        current_time = current_row[0]  # ego_data의 첫 번째 열이 'time'

        # 현재 줄을 포함한 30개 전 데이터 가져오기 (현재 줄 포함)
        if i - look_back >= 0:
            past_data = ego_data.iloc[i-look_back+1:i+1, 1:].values  # [x, y, velocity, yaw, x1, x2, x3] 사용
            
            # offset을 각 past_data에 따라 동적으로 생성
            offsets = np.array([[ego_data.iloc[i, 1], ego_data.iloc[i, 2], 0, 0, ego_data.iloc[j, 1], ego_data.iloc[j, 1], ego_data.iloc[j, 1]] for j in range(i-look_back+1, i+1)])
            past_data_flattened = past_data - offsets  # (30, 7) 배열과 element-wise 연산

            #past_data_flattened = past_data - offset  # (30, 7) 배열과 element-wise 연산
        else:
            continue  # 30개 전 데이터가 없는 경우 스킵

        # 30개 후 데이터 가져오기
        if i + look_ahead < len(ego_data):
            future_data = ego_data.iloc[i+1:i+1+look_ahead, [1, 2, 4]].values  # [x, y, yaw] 사용
            future_offset = np.array([ego_data.iloc[i, 1], ego_data.iloc[i, 2], 0])  # i번째의 [x, y, yaw]만 사용
            future_data_flattened = future_data - future_offset  # 해당 값만 빼줌
        else:
            continue  # 30개 후 데이터가 없는 경우 스킵

        # 결과 저장: past_data -> future_data 순으로 저장
        result_row = np.concatenate([past_data_flattened.flatten(), future_data_flattened.flatten()])
        result.append(result_row)

    return pd.DataFrame(result)

def load_and_save_ego_tracking_data(n, output_file):
    all_data = []  # 모든 데이터를 저장할 리스트

    for i in range(1, n + 1):
        file_name = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/tracking/ego_tracking_data_{i}.csv'  # 파일 경로 생성
        try:
            # CSV 파일 읽기
            data = pd.read_csv(file_name, header=None)  # 헤더가 없다고 가정
            all_data.append(data)  # 데이터를 리스트에 추가
            #print(f"Loaded data from {file_name} with {data.shape[0]} rows.")
        except FileNotFoundError:
            print(f"File {file_name} not found. Please check the file name and path.")

    # 모든 데이터를 하나의 DataFrame으로 결합
    combined_data = pd.concat(all_data, ignore_index=True)


    # 결합된 데이터를 새로운 CSV 파일로 저장
    combined_data.to_csv(output_file, index=False, header=False)  # 헤더 없이 저장
    print(f"Combined data saved to {output_file}")

    # # ego_tracking_data.csv 파일의 줄마다 처리
    # for i in range(look_back, len(ego_data) - look_ahead, skip_rows):
    #     current_row = ego_data.iloc[i]
    #     offset = [ego_data.iloc[i,1], ego_data.iloc[i,2], 0, 0, ego_data.iloc[i,1], ego_data.iloc[i,1], ego_data.iloc[i,1]] * look_back

    #     current_time = current_row[0]  # ego_data의 첫 번째 열이 'time'

    #     # 현재 줄을 포함한 30개 전 데이터 가져오기 (현재 줄 포함)
    #     if i - look_back >= 0:
    #         past_data = ego_data.iloc[i-look_back+1:i+1, 1:].values.flatten() - offset.values.flatten()  # [x, y, velocity, yaw, x1, x2, x3] 사용
    #     else:
    #         continue  # 30번째 전 데이터가 없는 경우 스킵

    #     # 30개 후 데이터 가져오기
    #     if i + look_ahead < len(ego_data):
    #         future_data = ego_data.iloc[i+1:i+1+look_ahead, [1, 2, 4]].values.flatten() - offset.values.flatten()  # [x, y, yaw] 사용
    #     else:
    #         continue  # 30번째 후 데이터가 없는 경우 스킵

    #     # 결과 저장: past_data -> future_data 순으로 저장
    #     result_row = np.concatenate([past_data, future_data])
    #     result.append(result_row)

    # return pd.DataFrame(result)

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
final_dataset.to_csv('/home/taewook/catkin_ws/src/beginner_tutorials/data/processed_data/dataset_ego.csv', index=False, header=False)
print("Saving sucess")
#final_dataset.to_csv('/home/taewook/catkin_ws/src/beginner_tutorials/data/processed_data/validation_ego.csv', index=False, header=False)

# raw 데이터 저장
output_file_path = '/home/taewook/catkin_ws/src/beginner_tutorials/data/ego_tracking_data_raw_combine.csv'
load_and_save_ego_tracking_data(index_finish, output_file_path)

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