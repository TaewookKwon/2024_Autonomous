import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

################
# File index
file_index = 17
## test1.bag으로 실행한 파일
################


# 파일 경로
ego_state_file = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/evaluation/ego_state_data_{file_index}.csv'
future_trajectory_file = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/evaluation/future_trajectory_{file_index}.csv'

# 데이터 로드
gt = pd.read_csv(ego_state_file, header=0).values
future_data = pd.read_csv(future_trajectory_file, header=0).values

# 초기 인덱스 설정
index = 0
duration = 31
start = 0
ade_steps = 31 # (1초까지의 ade를 계산)

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

# 초기 Figure와 Subplot 설정
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.ion()  # Interactive mode on

def update_plot():
    axs[0].clear()  # 첫 번째 서브플롯 초기화
    axs[1].clear()  # 두 번째 서브플롯 초기화
    
    time_gt = gt[index, 0]
    x_gt = gt[index:index + duration, 1]
    y_gt = gt[index:index + duration, 2]

    time_gt_int = (time_gt * 10**9).astype(int)
    future_indices = (future_data[:, 0] * 10**9).astype(int)
    future_index = np.where(future_indices == time_gt_int)[0]

    if future_index.size > 0:
        future_index = future_index[0]

        # 각 예측 모델의 경로 가져오기
        x_ctrv = future_data[future_index, 1:1 + duration * 2:2]
        y_ctrv = future_data[future_index, 2:2 + duration * 2:2]

        x_ctra = future_data[future_index, 1 + duration * 2:1 + duration * 4:2]
        y_ctra = future_data[future_index, 2 + duration * 2:2 + duration * 4:2]

        x_cv = future_data[future_index, 1 + duration * 4:1 + duration * 6:2]
        y_cv = future_data[future_index, 2 + duration * 4:2 + duration * 6:2]

        x_dynamics = future_data[future_index, 1 + duration * 6:1 + duration * 8:2]
        y_dynamics = future_data[future_index, 2 + duration * 6:2 + duration * 8:2]

        x_lstm = future_data[future_index, 1 + duration * 8:1 + duration * 10:2]
        y_lstm = future_data[future_index, 2 + duration * 8:2 + duration * 10:2]

        # ADE 계산
        ego_points = [(x_gt[i], y_gt[i]) for i in range(duration)]
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
        axs[0].plot(y_gt, x_gt, label='True', color='green', linewidth=3)  # Ground Truth
        axs[0].plot(y_ctrv, x_ctrv, label='CTRV Prediction', color='orange', linestyle='--')  # CTRV 모델
        axs[0].plot(y_ctra, x_ctra, label='CTRA Prediction', color='pink', linestyle='--')  # CTRA 모델
        axs[0].plot(y_cv, x_cv, label='CV Prediction', color='purple', linestyle='--')  # CV 모델
        axs[0].plot(y_lstm, x_lstm, label='Dynamics Prediction', color='blue', linestyle='--')
        axs[0].plot(y_dynamics, x_dynamics, label='LSTM Prediction', color='red', marker='s')  # LSTM 모델

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
    else:
        axs[0].title.set_text(f'No future data found for index: {index}')

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

def mean_ade(start, end):
    ade_ctrv_list = []
    ade_ctra_list = []
    ade_cv_list = []
    ade_dynamics_list = []
    ade_lstm_list = []

    for i in range(10,future_data.shape[0] - duration-20):
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
        ade_ctrv = calculate_ade(ego_points, list(zip(x_ctrv, y_ctrv)), start, end)
        ade_ctra = calculate_ade(ego_points, list(zip(x_ctra, y_ctra)), start, end)
        ade_cv = calculate_ade(ego_points, list(zip(x_cv, y_cv)), start, end)
        ade_dynamics = calculate_ade(ego_points, list(zip(x_dynamics, y_dynamics)), start, end)
        ade_lstm = calculate_ade(ego_points, list(zip(x_lstm, y_lstm)), start, end)

        # 각 시나리오의 ADE 결과를 리스트에 추가
        ade_ctrv_list.append(ade_ctrv)
        ade_ctra_list.append(ade_ctra)
        ade_cv_list.append(ade_cv)
        ade_dynamics_list.append(ade_dynamics)
        ade_lstm_list.append(ade_lstm)

    # 전체 시나리오에 대한 평균 계산
    ade_ctrv_mean = sum(ade_ctrv_list) / len(ade_ctrv_list)
    ade_ctra_mean = sum(ade_ctra_list) / len(ade_ctra_list)
    ade_cv_mean = sum(ade_cv_list) / len(ade_cv_list)
    ade_dynamics_mean = sum(ade_dynamics_list) / len(ade_dynamics_list)
    ade_lstm_mean = sum(ade_lstm_list) / len(ade_lstm_list)

    return ade_ctrv_mean, ade_ctra_mean, ade_cv_mean, ade_dynamics_mean, ade_lstm_mean

ade_start = 0
ade_end = duration
ade_ctrv_mean, ade_ctra_mean, ade_cv_mean, ade_dynamics_mean, ade_lstm_mean = mean_ade(ade_start, ade_end)

print("Average ADE in 1 scenario")
print(f"CTRV: {ade_ctrv_mean}")
print(f"CTRA: {ade_ctra_mean}")
print(f"CV: {ade_cv_mean}")
print(f"CTRV with dynamics: {ade_dynamics_mean}")
print(f"LSTM: {ade_lstm_mean}")

# 시간 별 결과 플롯 ##
# Tkinter 설정
root = tk.Tk()
root.title("Trajectory Visualization")
root.geometry("300x100")

# 버튼 설정
btn_prev = tk.Button(root, text="Previous", command=prev_index)
btn_prev.pack(side=tk.LEFT, padx=5, pady=5)

btn_next = tk.Button(root, text="Next", command=next_index)
btn_next.pack(side=tk.LEFT, padx=5, pady=5)

btn_set_index = tk.Button(root, text="Set Index", command=set_index)
btn_set_index.pack(side=tk.LEFT, padx=5, pady=5)

# 초기 플롯
update_plot()  # 초기 플롯을 업데이트하여 화면에 나타나게 합니다.

# Tkinter main loop
root.mainloop()


