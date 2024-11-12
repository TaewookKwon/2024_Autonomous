#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospkg

# 파일 경로 설정
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('beginner_tutorials')
full_path = pkg_path + '/path/lane3.txt'

# 파일 읽기
with open(full_path, 'r') as f:
    lines = f.readlines()

count = 1  # 줄 번호를 위한 카운터 변수

# 연속된 두 줄이 완전히 동일한지 확인
for i in range(len(lines) - 1):
    if lines[i].strip() == lines[i + 1].strip():  # strip()으로 공백 제거 후 비교
        print(f"True: Duplicate found at line {count} and {count + 1}")
        break  # 중복된 줄을 찾으면 루프 종료
    count += 1
else:
    print("False: No duplicate lines found.")