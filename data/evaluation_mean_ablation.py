def calculate_ade(ego_points, future_points, start, end):
    """Ego 차량의 좌표와 미래 예측 경로를 사용해 주어진 스텝 수까지 ADE를 계산합니다.
    
    Parameters:
    ego_points (list): Ego 차량의 포인트 리스트
    future_points (list): 예측된 포인트 리스트
    steps (int): 평균을 낼 포인트 수 (스텝 수)

    Returns:
    float: 평균 변위 오차(ADE)
    """
    # 평균을 낼 스텝 수만큼 포인트를 자릅니다.
    ego_points = ego_points[start:end]
    future_points = future_points[start:end]

    # 각 포인트 쌍의 오차 계산
    errors = [
        np.linalg.norm(np.array(ego) - np.array(future))
        for ego, future in zip(ego_points, future_points)
    ]
    
    return np.mean(errors)

def calculate_fde(ego_points, future_points):
    """Ego 차량의 최종 위치와 미래 예측 경로의 최종 위치 사이의 FDE를 계산합니다.
    
    Parameters:
    ego_points (list): Ego 차량의 포인트 리스트
    future_points (list): 예측된 포인트 리스트

    Returns:
    float: 최종 변위 오차(FDE)
    """
    # 마지막 포인트 추출
    ego_final_point = ego_points[-1]  # 마지막 에고 포인트
    future_final_point = future_points[-1]  # 마지막 예측 포인트

    # 두 점 간의 거리 계산
    fde = np.linalg.norm(np.array(ego_final_point) - np.array(future_final_point))

    return fde

def update_plot():
    axs[0].clear()  # 첫 번째 서브플롯 초기화
    axs[1].clear()  # 두 번째 서브플롯 초기화
    
    time_gt = gt[index, 0]
    x_gt = gt[index:index + duration, 1]
    y_gt = gt[index:index + duration, 2]

    time_gt_int = (time_gt * 10**9).astype(int)
    future_indices = (future_data[:, 0] * 10**9).astype(int)
    future_index = np.where(future_indices == time_gt_int)[0]

    # if future_index.size > 0:
    #     future_index = future_index[0]

    # 각 예측 모델의 경로 가져오기
    x_ctrv = future_data[index, 1:1 + duration * 2 :2]
    y_ctrv = future_data[index, 2:2 + duration * 2 :2]

    x_ctra = future_data[index, 1 + duration * 2:1 + duration * 4 :2]
    y_ctra = future_data[index, 2 + duration * 2:2 + duration * 4 :2]

    x_cv = future_data[index, 1 + duration * 4:1 + duration * 6 :2]
    y_cv = future_data[index, 2 + duration * 4:2 + duration * 6 :2]

    x_dynamics = future_data[index, 1 + duration * 6:1 + duration * 8 :2]
    y_dynamics = future_data[index, 2 + duration * 6:2 + duration * 8 :2]

    x_lstm = future_data[index, 1 + duration * 8:1 + duration * 10 :2]
    y_lstm = future_data[index, 2 + duration * 8:2 + duration * 10 :2]

    # ADE 계산
    ego_points = [(x_gt[i], y_gt[i]) for i in range(duration-1)]
    ade_ctrv = calculate_ade(ego_points, list(zip(x_ctrv, y_ctrv)), start, ade_steps)
    ade_ctra = calculate_ade(ego_points, list(zip(x_ctra, y_ctra)), start, ade_steps)
    ade_cv = calculate_ade(ego_points, list(zip(x_cv, y_cv)), start, ade_steps)
    ade_dynamics = calculate_ade(ego_points, list(zip(x_dynamics, y_dynamics)), start, ade_steps)
    ade_lstm = calculate_ade(ego_points, list(zip(x_lstm, y_lstm)), start, ade_steps)

    # 첫 번째 서브플롯: Trajectory Plot
    # axs[0].plot(x_gt, y_gt, label='True', color='green', alpha=0.5)  # Ground Truth
    # axs[0].plot(x_ctrv, y_ctrv, label='CTRV Prediction', color='orange', linestyle='--')  # CTRV 모델
    # axs[0].plot(x_ctra, y_ctra, label='CTRA Prediction', color='yellow', linestyle='--')  # CTRA 모델
    # axs[0].plot(x_cv, y_cv, label='CV Prediction', color='purple', linestyle='--')  # CV 모델
    # axs[0].plot(x_lstm, y_lstm, label='LSTM Prediction', color='red', marker='s')  # LSTM 모델
    axs[0].plot(y_gt, x_gt, label='True', color='green', linewidth=3,marker='o')  # Ground Truth
    axs[0].plot(y_ctrv, x_ctrv, label='CTRV Prediction', color='orange', linestyle='--')  # CTRV 모델
    axs[0].plot(y_ctra, x_ctra, label='CTRA Prediction', color='pink', linestyle='--')  # CTRA 모델
    axs[0].plot(y_cv, x_cv, label='CV Prediction', color='purple', linestyle='--')  # CV 모델
    axs[0].plot(y_dynamics, x_dynamics, label='Dynamics Prediction', color='blue', linestyle='--')
    axs[0].plot(y_lstm, x_lstm, label='LSTM Prediction', color='red', marker='s')  # LSTM 모델

    axs[0].set_title(f'Trajectory Comparison at Time: {round(index * 0.1, 2)}s')
    axs[0].legend()
    axs[0].grid()
    axs[0].axis('equal')
    axs[0].invert_xaxis()  # x축 반전
    axs[0].set_ylim(212, 204)  # y축 범위 설정

    # 두 번째 서브플롯: ADE Plot
    ade_values = [ade_ctrv, ade_ctra, ade_cv, ade_dynamics, ade_lstm]
    axs[1].bar(['CTRV', 'CTRA', 'CV', 'Dynamics', 'LSTM'], ade_values, color=['orange', 'pink', 'purple', 'blue', 'red'])
    axs[1].set_ylabel('ADE')
    axs[1].set_title(f'Average Displacement Error (ADE) for Predictions in {int(ade_steps*0.1)} sec')
    axs[1].grid(True)
    # else:
    #     axs[0].title.set_text(f'No future data found for index: {index}')

    plt.tight_layout()  # 레이아웃 자동 조정
    plt.draw()  # 플롯을 업데이트합니다.
    plt.pause(0.1)  # 잠시 대기하여 플롯이 화면에 나타나도록 합니다.

def next_index():
    global index
    if index < len(gt) - duration - 1:
        index += 1
    update_plot()

def prev_index():
    global index
    if index > 0:
        index -= 1
    update_plot()

def set_index():
    global index
    new_index = simpledialog.askinteger("Set Index", "Enter index (0 to {}):".format(len(gt) - duration - 1))
    if new_index is not None and 0 <= new_index < len(gt) - duration:
        index = new_index
        update_plot()

def mean_ade(gt, future_data, ade_start, ade_end):
    duration = 31
    ade_ctrv_list = []
    ade_ctra_list = []
    ade_cv_list = []
    ade_dynamics_list = []
    ade_lstm_list = []

    for i in range(0, future_data.shape[0] - duration):
        x_gt = gt[i:i + duration, 1]
        y_gt = gt[i:i + duration, 2]

        # 각 예측 모델의 경로 가져오기
        x_ctrv = future_data[i, 1:1 + duration * 2:2]
        y_ctrv = future_data[i, 2:2 + duration * 2:2]
        
        x_ctra = future_data[i, 1 + duration * 2:1 + duration * 4:2]
        y_ctra = future_data[i, 2 + duration * 2:2 + duration * 4:2]

        x_cv = future_data[i, 1 + duration * 4:1 + duration * 6:2]
        y_cv = future_data[i, 2 + duration * 4:2 + duration * 6:2]

        x_dynamics = future_data[i, 1 + duration * 6:1 + duration * 8:2]
        y_dynamics = future_data[i, 2 + duration * 6:2 + duration * 8:2]

        x_lstm = future_data[i, 1 + duration * 8:1 + duration * 10:2]
        y_lstm = future_data[i, 2 + duration * 8:2 + duration * 10:2]

        # ADE 계산
        ego_points = [(x_gt[j], y_gt[j]) for j in range(duration)]
        ade_ctrv = calculate_ade(ego_points, list(zip(x_ctrv, y_ctrv)), ade_start, ade_end)
        ade_ctra = calculate_ade(ego_points, list(zip(x_ctra, y_ctra)), ade_start, ade_end)
        ade_cv = calculate_ade(ego_points, list(zip(x_cv, y_cv)), ade_start, ade_end)
        ade_dynamics = calculate_ade(ego_points, list(zip(x_dynamics, y_dynamics)), ade_start, ade_end)
        ade_lstm = calculate_ade(ego_points, list(zip(x_lstm, y_lstm)), ade_start, ade_end)


        # 각 시나리오의 ADE 결과를 리스트에 추가
        ade_ctrv_list.append(ade_ctrv)
        ade_ctra_list.append(ade_ctra)
        ade_cv_list.append(ade_cv)
        ade_dynamics_list.append(ade_dynamics)
        ade_lstm_list.append(ade_lstm)

    # 각 시나리오에 대한 평균 계산
    ade_ctrv_mean = sum(ade_ctrv_list) / len(ade_ctrv_list)
    ade_ctra_mean = sum(ade_ctra_list) / len(ade_ctra_list)
    ade_cv_mean = sum(ade_cv_list) / len(ade_cv_list)
    ade_dynamics_mean = sum(ade_dynamics_list) / len(ade_dynamics_list)
    ade_lstm_mean = sum(ade_lstm_list) / len(ade_lstm_list)

    return ade_ctrv_mean, ade_ctra_mean, ade_cv_mean, ade_dynamics_mean, ade_lstm_mean

def mean_fde(gt, future_data):
    duration = 31
    fde_ctrv_list = []
    fde_ctra_list = []
    fde_cv_list = []
    fde_dynamics_list = []
    fde_lstm_list = []

    for i in range(0, future_data.shape[0] - duration):
        x_gt = gt[i:i + duration, 1]
        y_gt = gt[i:i + duration, 2]

        # 각 예측 모델의 경로 가져오기
        x_ctrv = future_data[i, 1:1 + duration * 2:2]
        y_ctrv = future_data[i, 2:2 + duration * 2:2]
        
        x_ctra = future_data[i, 1 + duration * 2:1 + duration * 4:2]
        y_ctra = future_data[i, 2 + duration * 2:2 + duration * 4:2]

        x_cv = future_data[i, 1 + duration * 4:1 + duration * 6:2]
        y_cv = future_data[i, 2 + duration * 4:2 + duration * 6:2]

        x_dynamics = future_data[i, 1 + duration * 6:1 + duration * 8:2]
        y_dynamics = future_data[i, 2 + duration * 6:2 + duration * 8:2]

        x_lstm = future_data[i, 1 + duration * 8:1 + duration * 10:2]
        y_lstm = future_data[i, 2 + duration * 8:2 + duration * 10:2]

        # FDE 계산
        ego_points = [(x_gt[j], y_gt[j]) for j in range(duration)]
        fde_ctrv = calculate_fde(ego_points, list(zip(x_ctrv, y_ctrv)))
        fde_ctra = calculate_fde(ego_points, list(zip(x_ctra, y_ctra)))
        fde_cv = calculate_fde(ego_points, list(zip(x_cv, y_cv)))
        fde_dynamics = calculate_fde(ego_points, list(zip(x_dynamics, y_dynamics)))
        fde_lstm = calculate_fde(ego_points, list(zip(x_lstm, y_lstm)))

        # 각 시나리오의 FDE 결과를 리스트에 추가
        fde_ctrv_list.append(fde_ctrv)
        fde_ctra_list.append(fde_ctra)
        fde_cv_list.append(fde_cv)
        fde_dynamics_list.append(fde_dynamics)
        fde_lstm_list.append(fde_lstm)

    # 각 시나리오에 대한 평균 계산
    fde_ctrv_mean = sum(fde_ctrv_list) / len(fde_ctrv_list)
    fde_ctra_mean = sum(fde_ctra_list) / len(fde_ctra_list)
    fde_cv_mean = sum(fde_cv_list) / len(fde_cv_list)
    fde_dynamics_mean = sum(fde_dynamics_list) / len(fde_dynamics_list)
    fde_lstm_mean = sum(fde_lstm_list) / len(fde_lstm_list)

    return fde_ctrv_mean, fde_ctra_mean, fde_cv_mean, fde_dynamics_mean, fde_lstm_mean


def simulate(ade_start, ade_end,index_list):
    global entire_sum_ade_ctrv, entire_sum_ade_ctra, entire_sum_ade_cv, entire_sum_ade_dynamics, entire_sum_ade_lstm
    global entire_sum_fde_ctrv, entire_sum_fde_ctra, entire_sum_fde_cv, entire_sum_fde_dynamics, entire_sum_fde_lstm
    
    # 각 ADE와 FDE의 전체 평균을 저장할 리스트 초기화
    entire_sum_ade_ctrv, entire_sum_ade_ctra, entire_sum_ade_cv, entire_sum_ade_dynamics, entire_sum_ade_lstm = [], [], [], [], []
    entire_sum_fde_ctrv, entire_sum_fde_ctra, entire_sum_fde_cv, entire_sum_fde_dynamics, entire_sum_fde_lstm = [], [], [], [], []
    
    for file_index in index_list:
        # 파일 경로
        ego_state_file = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/evaluation/ego_state_data_{file_index}.csv'
        future_trajectory_file = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/evaluation/future_trajectory_{file_index}.csv'

        # 데이터 로드
        gt = pd.read_csv(ego_state_file, header=0).values
        future_data = pd.read_csv(future_trajectory_file, header=0).values

        # 각 파일에 대한 평균 ADE 및 FDE 계산
        ade_ctrv_mean, ade_ctra_mean, ade_cv_mean, ade_dynamics_mean, ade_lstm_mean = mean_ade(gt, future_data, ade_start, ade_end)
        fde_ctrv_mean, fde_ctra_mean, fde_cv_mean, fde_dynamics_mean, fde_lstm_mean = mean_fde(gt, future_data)

        # 전체 평균을 저장 (ADE와 FDE)
        entire_sum_ade_ctrv.append(ade_ctrv_mean)
        entire_sum_ade_ctra.append(ade_ctra_mean)
        entire_sum_ade_cv.append(ade_cv_mean)
        entire_sum_ade_dynamics.append(ade_dynamics_mean)
        entire_sum_ade_lstm.append(ade_lstm_mean)

        entire_sum_fde_ctrv.append(fde_ctrv_mean)
        entire_sum_fde_ctra.append(fde_ctra_mean)
        entire_sum_fde_cv.append(fde_cv_mean)
        entire_sum_fde_dynamics.append(fde_dynamics_mean)
        entire_sum_fde_lstm.append(fde_lstm_mean)

        print(f"From {int(ade_start * 0.1)} sec to {int(ade_end * 0.1)} sec")
        print(f"Average ADE in Scenario {file_index}")
        print(f"CTRV: {ade_ctrv_mean}")
        print(f"CTRA: {ade_ctra_mean}")
        print(f"CV: {ade_cv_mean}")
        print(f"LSTM with Lane data: {ade_dynamics_mean}")
        print(f"LSTM: {ade_lstm_mean}")
        print()
        print(f"Average FDE in Scenario {file_index}")
        print(f"CTRV: {fde_ctrv_mean}")
        print(f"CTRA: {fde_ctra_mean}")
        print(f"CV: {fde_cv_mean}")
        print(f"LSTM with Lane data: {fde_dynamics_mean}")
        print(f"LSTM: {fde_lstm_mean}")
        print("---------------------------")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

pd.set_option('display.max_seq_items', None)

################
# # File index
index_list = [14, 15, 16, 17, 22]
################

## pth_4 는 2,3,4,5,6

### 평균 ADE 관찰시간을 조정
ade_start = 11
ade_end = 31

# ADE와 FDE 결과를 저장할 리스트 초기화
entire_sum_ade_ctrv = []
entire_sum_ade_ctra = []
entire_sum_ade_cv = []
entire_sum_ade_dynamics = []
entire_sum_ade_lstm = []

entire_sum_fde_ctrv = []
entire_sum_fde_ctra = []
entire_sum_fde_cv = []
entire_sum_fde_dynamics = []
entire_sum_fde_lstm = []

# simulate 함수 호출
simulate(ade_start, ade_end, index_list)

# 시나리오 전체 평균을 계산 (ADE)
entire_mean_ade_ctrv = sum(entire_sum_ade_ctrv) / len(entire_sum_ade_ctrv)
entire_mean_ade_ctra = sum(entire_sum_ade_ctra) / len(entire_sum_ade_ctra)
entire_mean_ade_cv = sum(entire_sum_ade_cv) / len(entire_sum_ade_cv)
entire_mean_ade_dynamics = sum(entire_sum_ade_dynamics) / len(entire_sum_ade_dynamics)
entire_mean_ade_lstm = sum(entire_sum_ade_lstm) / len(entire_sum_ade_lstm)

print("Average ADE in Entire Scenario:")
print(f"CTRV: {entire_mean_ade_ctrv}")
print(f"CTRA: {entire_mean_ade_ctra}")
print(f"CV: {entire_mean_ade_cv}")
print(f"LSTM with Lane data: {entire_mean_ade_dynamics}")
print(f"LSTM: {entire_mean_ade_lstm}")

# 시나리오 전체 평균을 계산 (FDE)
entire_mean_fde_ctrv = sum(entire_sum_fde_ctrv) / len(entire_sum_fde_ctrv)
entire_mean_fde_ctra = sum(entire_sum_fde_ctra) / len(entire_sum_fde_ctra)
entire_mean_fde_cv = sum(entire_sum_fde_cv) / len(entire_sum_fde_cv)
entire_mean_fde_dynamics = sum(entire_sum_fde_dynamics) / len(entire_sum_fde_dynamics)
entire_mean_fde_lstm = sum(entire_sum_fde_lstm) / len(entire_sum_fde_lstm)

print("Average FDE in Entire Scenario:")
print(f"CTRV: {entire_mean_fde_ctrv}")
print(f"CTRA: {entire_mean_fde_ctra}")
print(f"CV: {entire_mean_fde_cv}")
print(f"LSTM with Lane data: {entire_mean_fde_dynamics}")
print(f"LSTM: {entire_mean_fde_lstm}")


