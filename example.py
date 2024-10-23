import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Chuỗi thời gian của chart chính (crypto A)
time_series_a = np.array([100, 102, 101, 103, 104, 105, 107])

# Chuỗi thời gian của các chart khác (crypto B, C, D, ...)
time_series_b = np.array([99, 101, 100, 102, 104, 106, 108])
time_series_c = np.array([101, 103, 102, 104, 105, 106, 109])

# So sánh chuỗi thời gian A với B
distance_b, path_b = fastdtw(time_series_a, time_series_b, dist=euclidean)

# So sánh chuỗi thời gian A với C
distance_c, path_c = fastdtw(time_series_a, time_series_c, dist=euclidean)

# In khoảng cách giữa chuỗi A và các chuỗi khác
print(f"Distance between A and B: {distance_b}")
print(f"Distance between A and C: {distance_c}")

# Xếp hạng sự tương đồng
if distance_b < distance_c:
    print("Crypto B có hành vi giá tương tự với A")
else:
    print("Crypto C có hành vi giá tương tự với A")
