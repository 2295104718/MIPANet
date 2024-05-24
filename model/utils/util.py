import numpy as np
def get_param_ids(module_list):
    param_ids = []
    for mo in module_list:
        ids = list(map(id, mo.parameters()))
        param_ids = param_ids + ids
    return param_ids

class_colors = [(0, 0, 0),
                    # 0=background
                    (148, 65, 137), (255, 116, 69), (86, 156, 137),
                    (202, 179, 158), (155, 99, 235), (161, 107, 108),
                    (133, 160, 103), (76, 152, 126), (84, 62, 35),
                    (44, 80, 130), (31, 184, 157), (101, 144, 77),
                    (23, 197, 62), (141, 168, 145), (142, 151, 136),
                    (115, 201, 77), (100, 216, 255), (57, 156, 36),
                    (88, 108, 129), (105, 129, 112), (42, 137, 126),
                    (155, 108, 249), (166, 148, 143), (81, 91, 87),
                    (100, 124, 51), (73, 131, 121), (157, 210, 220),
                    (134, 181, 60), (221, 223, 147), (123, 108, 131),
                    (161, 66, 179), (163, 221, 160), (31, 146, 98),
                    (99, 121, 30), (49, 89, 240), (116, 108, 9),
                    (161, 176, 169), (80, 29, 135), (177, 105, 197),
                    (139, 110, 246)]

CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
                    (137, 28, 157), (150, 255, 255), (54, 114, 113),
                    (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
                    (255, 150, 255), (255, 180, 10), (101, 70, 86),
                    (38, 230, 0), (255, 120, 70), (117, 41, 121),
                    (150, 255, 0), (132, 0, 255), (24, 209, 255),
                    (191, 130, 35), (219, 200, 109), (154, 62, 86),
                    (255, 190, 190), (255, 0, 255), (192, 79, 212),
                    (152, 163, 55), (230, 230, 230), (53, 130, 64),
                    (155, 249, 152), (87, 64, 34), (214, 209, 175),
                    (170, 0, 59), (255, 0, 0), (193, 195, 234), (70, 72, 115),
                    (255, 255, 0), (52, 57, 131), (12, 83, 45)]

# def color_label_eval(label):
#     # label = label.clone().cpu().data.numpy()
#     colored_label = np.vectorize(lambda x: CLASS_COLORS[int(x)])
#
#     colored = np.asarray(colored_label(label)).astype(np.float32)
#     colored = colored.squeeze()
#
#     # return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
#     return colored.transpose([0, 2, 1])

def color_label_eval(label):
    # 创建一个形状为[高度, 宽度, 3]的数组，用于存储RGB值
    colored = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    for class_idx, color in enumerate(CLASS_COLORS):
        colored[label == class_idx] = color  # 将相应类别的位置设置为对应的颜色

    return colored

