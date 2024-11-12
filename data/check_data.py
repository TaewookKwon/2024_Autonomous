import os

index = 2
length = 1 + 31*2 * 5

# CSV 파일을 한 줄씩 읽고, 각 줄의 쉼표로 구분된 값의 개수가 183개인지 확인하는 코드
def check_csv_file_line_lengths(file_path, expected_length=length):
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, start=1):
            values = line.strip().split(',')
            if len(values) != expected_length:
                return f"Failure: Line {line_num} has {len(values)} values instead of {expected_length}."
    return "Success: All lines have the correct number of values."
    #return len(values)

# 파일 경로 설정
file_path = f'/home/taewook/catkin_ws/src/beginner_tutorials/data/evaluation/future_trajectory_{index}.csv'

# 결과 출력
check_result = check_csv_file_line_lengths(file_path)
print(check_result)
# print(f"{check_result} Data")