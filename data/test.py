import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Figure와 Axes 생성
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# 버튼 추가
ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
btn_next = Button(ax_next, 'Next')

def next_index(event):
    print("Next button clicked!")  # 버튼 클릭 시 동작
btn_next.on_clicked(next_index)

plt.show()  # 이벤트 루프 시작