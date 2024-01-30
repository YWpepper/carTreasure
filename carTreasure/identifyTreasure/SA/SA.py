import math
import random
from TFS import get_way
from TFS import simulate
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# t0为初始温度，t_final为终止温度，alpha为冷却系数，inner_iter为内层迭代次数
# city_numbers为所需要经过的其他城市，length_mat为各个节点之间的最短距离矩阵
def sa(t0, t_final, alpha, inner_iter, city_numbers, po_list_t, map_int, start_x, start_y, target_x, target_y):
    all_path = []
    all_ex = []
    best_way = ''
    init = initialization(city_numbers)
    all_path.append(init)
    all_ex.append(e(init, po_list_t, map_int, city_numbers, start_x, start_y, target_x, target_y)[0])
    t = t0
    while t > t_final:
        path = search(all_path, po_list_t, map_int, city_numbers, start_x, start_y, target_x, target_y)
        ex, way = e(path, po_list_t, map_int, city_numbers, start_x, start_y, target_x, target_y)
        for i in range(inner_iter):
            new_path = generate_new(path)
            new_ex, way = e(new_path, po_list_t, map_int, city_numbers, start_x, start_y, target_x, target_y)
            if metropolis(ex, new_ex, t):
                path = new_path
                ex = new_ex
                best_way = way
        all_path.append(path)
        all_ex.append(ex)
        draw(all_path, all_ex)
        simulate(start_x, start_y, best_way, po_list_t, map_int)
        t = alpha * t
        print(t, ex)
    return all_path, all_ex, best_way


def initialization(numbers):
    path_random = np.random.choice(list(range(0, numbers)), replace=False, size=numbers)
    path_random = path_random.tolist()
    print(path_random)
    return path_random


def generate_new(path):
    numbers = len(path)
    # 随机生成两个不重复的点
    positions = np.random.choice(list(range(numbers)), replace=False, size=2)
    lower_position = min(positions[0], positions[1])
    upper_position = max(positions[0], positions[1])
    # 将数列中段逆转
    mid_reversed = path[lower_position:upper_position + 1]
    mid_reversed.reverse()
    # 拼接生成新的数列
    new_path = path[:lower_position]
    new_path.extend(mid_reversed)
    new_path.extend(path[upper_position + 1:])
    return new_path


def e(case_num, po_list_t, map_int, num, start_x, start_y, target_x, target_y):
    total_way = ''
    cnt = 0
    for i in case_num:
        liter_point = po_list_t[i]
        cnt = cnt + 1
        step_way = ''
        if cnt == 1:
            step_way = get_way(start_x, start_y, liter_point[0], liter_point[1], map_int)
        elif 1 < cnt <= num:
            step_way = get_way(last_point[0], last_point[1], liter_point[0], liter_point[1], map_int)
        last_point = liter_point
        step_way = step_way + "#"
        total_way += step_way
        if cnt == num:
            end_way = get_way(liter_point[0], liter_point[1], target_x, target_y, map_int)
            total_way += end_way
    return len(total_way) - num, total_way


def metropolis(_e, new_e, t):
    if new_e <= _e:
        return True
    else:
        p = math.exp((_e - new_e) / t)
        return True if random.random() < p else False


def search(all_path, po_list_t, map_int, num, start_x, start_y, target_x, target_y):
    best_e = 0xffff
    best_path = all_path[0]
    for path in all_path:
        ex, way = e(path, po_list_t, map_int, num, start_x, start_y, target_x, target_y)
        if ex < best_e:
            best_e = ex
            best_path = path
    return best_path


def draw(all_path, all_ex):
    
    plt.figure(1)
    iteration = len(all_path)
    all_ex = np.array(all_ex, dtype="object")
    plt.xlabel("Iteration")
    plt.ylabel("Length")
    plt.plot(range(iteration), all_ex, color='blue')
    plt.pause(0.1)
    

# def draw(all_path, all_ex):

#     save_path = './'

#     iteration = len(all_path)
#     all_ex = np.array(all_ex, dtype="object")

#     # 创建一个黑色背景的图像
#     img = Image.new('RGB', (600, 400), (0, 0, 0))
#     draw = ImageDraw.Draw(img)

#     # 绘制坐标轴
#     draw.line((50, 350, 550, 350), fill=(255, 255, 255), width=2)
#     draw.line((50, 350, 50, 50), fill=(255, 255, 255), width=2)

#     # 绘制数据曲线
#     for i in range(iteration - 1):
#         p1 = (50 + int(500 * i / (iteration - 1)), 350 - int(all_ex[i] * 200))
#         p2 = (50 + int(500 * (i + 1) / (iteration - 1)), 350 - int(all_ex[i + 1] * 200))
#         draw.line((p1, p2), fill=(0, 0, 255), width=2)

#     # 保存图像
#     img.save(save_path)
