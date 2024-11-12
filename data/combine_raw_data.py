import pandas as pd

def load_and_save_ego_tracking_data(n, output_file):
    all_data = []  # 모든 데이터를 저장할 리스트

    for i in range(1, n + 1):
        file_name = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/tracking/ego_tracking_data_{i}.csv'  # 파일 경로 생성
        try:
            # CSV 파일 읽기
            data = pd.read_csv(file_name, header=None)  # 헤더가 없다고 가정
            all_data.append(data)  # 데이터를 리스트에 추가
            print(f"Loaded data from {file_name} with {data.shape[0]} rows.")
        except FileNotFoundError:
            print(f"File {file_name} not found. Please check the file name and path.")

    # 모든 데이터를 하나의 DataFrame으로 결합
    combined_data = pd.concat(all_data, ignore_index=True)


    # 결합된 데이터를 새로운 CSV 파일로 저장
    combined_data.to_csv(output_file, index=False, header=False)  # 헤더 없이 저장
    print(f"Combined data saved to {output_file}")


# 사용 예시
n = 36  # 불러올 파일의 개수
output_file_path = '/home/taewook/catkin_ws/src/beginner_tutorials/data/ego_tracking_data_raw_combine.csv'
load_and_save_ego_tracking_data(n, output_file_path)