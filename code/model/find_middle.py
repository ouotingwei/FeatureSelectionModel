import numpy as np

array=np.load('/home/lab605/deep_feature_selection/training_data/small_coffee/2/error.npy')
# 創建一個隨機的浮點數一維數組
# array = np.random.rand(10)

# 將數組排序
sorted_array = np.sort(array)

# 打印排序後的數組
print("排序後的數組:", sorted_array)

# 找出中間值
length = len(sorted_array)
if length % 2 == 0:
    middle_value = (sorted_array[length // 2 - 1] + sorted_array[length // 2]) / 2
else:
    middle_value = sorted_array[length // 2]

print("中間值:", middle_value)
