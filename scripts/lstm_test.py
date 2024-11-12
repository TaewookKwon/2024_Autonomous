import torch
import torch.nn as nn
from beginner_tutorials.msg import TrajectoryArray, TrajectoryPoint  # 실제 메시지 파일 경로를 맞게 수정
import pandas as pd
import matplotlib.pyplot as plt

# Define the LSTM model
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        self.device = device

    def forward(self, x, num_layers=2, hidden_size=128):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(self.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(self.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

def load_lstm_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrajectoryLSTM(input_size=7, hidden_size=128, num_layers=2, output_size=90, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_motion_LSTM(input_tensor):
    with torch.no_grad():
        predicted_trajectory = lstm_model(input_tensor)
    
    predicted_trajectory = predicted_trajectory.view(30, 3).cpu().numpy()
    trajectory_array = TrajectoryArray()
    for t in range(30):
        point = TrajectoryPoint()
        point.x = predicted_trajectory[t, 0]
        point.y = predicted_trajectory[t, 1]
        point.yaw = predicted_trajectory[t, 2]
        point.time = t * 0.1
        trajectory_array.points.append(point)

    return trajectory_array

# 평균, 표준편차 불러오기
raw_file_path = '/home/taewook/catkin_ws/src/beginner_tutorials/data/ego_tracking_data_raw_combine.csv'
raw_data = pd.read_csv(raw_file_path, header=None)  # 헤더가 없다고 가정
data_mean = raw_data.mean().values[1:].tolist()  # 1부터 끝까지 선택하여 저장
data_std = raw_data.std().values[1:].tolist()    # 1부터 끝까지 선택하여 저장

# Load the model
model_path = '/home/taewook/LSTM/trajectory_lstm_model_2.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = load_lstm_model(model_path)

# Load data
path = '/home/taewook/catkin_ws/src/beginner_tutorials/data/dataset1.csv'
future_data = pd.read_csv(path, header=None).values

input1 = future_data[10][:210]
input2 = future_data[170][:210]

input1_tensor = torch.tensor(input1, dtype=torch.float32).view(1, 30, 7).to(device)
input2_tensor = torch.tensor(input2, dtype=torch.float32).view(1, 30, 7).to(device)

output1 = predict_motion_LSTM(input1_tensor)
output2 = predict_motion_LSTM(input2_tensor)

# Prepare data for plotting
x_values_1 = [(point.x * data_std[0]) + data_mean[0] for point in output1.points]
y_values_1 = [(point.y * data_std[1]) + data_mean[1] for point in output1.points]

x_values_2 = [(point.x * data_std[0]) + data_mean[0] for point in output2.points]
y_values_2 = [(point.y * data_std[1]) + data_mean[1] for point in output2.points]

# Plot the results
plt.plot(x_values_1, y_values_1, label='Data1', color='blue')
plt.plot(x_values_2, y_values_2, label='Data2', color='yellow')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Predicted Trajectories')
plt.legend()

# Set equal scaling for both axes
plt.axis('equal')  # 또는 plt.gca().set_aspect('equal', adjustable='box') 사용 가능

plt.show()  # Ensure to call show() to display the plot

