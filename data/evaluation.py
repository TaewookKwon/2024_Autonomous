import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv_data(file_path):
    """CSV 파일을 로드하고 내용을 리스트로 반환합니다."""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 건너뛰기
        return [list(map(float, row)) for row in reader]

def calculate_ade(ego_points, future_points):
    """Ego 차량의 좌표와 미래 예측 경로를 사용해 ADE를 계산합니다."""
    errors = [
        np.linalg.norm(np.array(ego) - np.array(future))
        for ego, future in zip(ego_points, future_points)
    ]
    return np.mean(errors)

def visualize_trajectories(ego_points, future_points, future_time):
    """Ego와 예측 경로를 시각화합니다."""
    ego_x, ego_y = zip(*ego_points)
    future_x, future_y = zip(*future_points)

    plt.figure(figsize=(8, 6))
    plt.plot(ego_x, ego_y, marker='o', linestyle='-', color='blue', label='Ego Trajectory')
    plt.plot(future_x, future_y, marker='x', linestyle='--', color='red', label='Future Prediction')
    plt.title(f'Trajectory Comparison at Time {future_time:.2f}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # CSV 파일 불러오기
    ego_data = load_csv_data('ego_state_data.csv')
    future_data = load_csv_data('future_trajectory.csv')

    # ego 데이터의 시간값과 좌표를 딕셔너리에 저장 (빠른 접근용)
    ego_time_index = {row[0]: (row[1], row[2]) for row in ego_data}

    # 결과를 저장할 리스트
    times = []
    ade_values = []

    for future_line in future_data:
        future_time = future_line[0]

        # 미래 경로에서 (x1, y1), (x2, y2), ..., (x30, y30) 추출
        future_points = [
            (future_line[i], future_line[i + 1])
            for i in range(1, len(future_line), 2)
        ]

        # ego 데이터에서 future_time과 일치하는 행을 찾고, 그 시점부터 30개 점 추출
        ego_points = []
        for i in range(30):
            time_key = round(future_time + i * 0.1, 2)  # 소수점 2자리까지 반올림

            # 해당 시간에 대한 ego 좌표를 딕셔너리에서 찾음
            if time_key in ego_time_index:
                ego_points.append(ego_time_index[time_key])
            else:
                print(f"Missing ego data for time {time_key}")
                return  # 데이터 누락 시 계산 중단

        # ADE 계산
        ade = calculate_ade(ego_points, future_points)

        # 시간과 ADE 결과 저장
        times.append(future_time)
        ade_values.append(ade)

        # Trajectory 시각화
        visualize_trajectories(ego_points, future_points, future_time)

    # ADE 그래프 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(times, ade_values, marker='o', linestyle='-', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('ADE')
    plt.title('Average Displacement Error (ADE) Over Time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
