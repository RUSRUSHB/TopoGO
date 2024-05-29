# import numpy as np

# edge_labels = [3,1,2,1]
# edge_labels_shift = [1,2,1,3]

# unique_labels = np.unique(edge_labels)

# up_label, down_label_1, down_label_2 = -1, -1, -1
# # 查看edge_labels和edge_labels_shift中是否有出现了三次的标签
# for label in unique_labels:
#     if edge_labels.count(label) == 3 or edge_labels_shift.count(label) == 3:
#         up_label = label
#         break
#     if edge_labels.count(label) == 2 and edge_labels_shift.count(label) == 2:
#         up_label = label
#         break

# if up_label == -1:
#     print("No crossing")
#     exit()

# [down_label_1, down_label_2] = [x for x in unique_labels if x != up_label]

# print(up_label, down_label_1, down_label_2)

# import numpy as np

# window = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# edge_labels = np.concatenate([window[0:-1, 0], window[-1, :], window[-2:0:-1, -1], window[0, -1:0:-1]])

# print(edge_labels)

import numpy as np

edge_labels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(edge_labels)
print(edge_labels.tolist())