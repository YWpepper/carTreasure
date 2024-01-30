import itertools
import matplotlib.pyplot as plt
import time

T1 = time.time()


class Node(object):
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

    def __str__(self):
        return self.w


def up(node):
    return Node(node.x - 1, node.y, node.w + "L")


def down(node):
    return Node(node.x + 1, node.y, node.w + "R")


def left(node):
    return Node(node.x, node.y - 1, node.w + "U")


def right(node):
    return Node(node.x, node.y + 1, node.w + "D")


def get_way(start_x, start_y, target_x, target_y, map_int):
    m = len(map_int[0])
    n = len(map_int)

    queue = []
    visited = []

    node = Node(start_x, start_y, "")
    queue.append(node)
    while len(queue) != 0:
        moveNode = queue[0]
        queue.pop(0)
        moveStr = str(moveNode.x) + " " + str(moveNode.y)
        if moveStr not in visited:
            visited.append(moveStr)
            if moveNode.x == target_x and moveNode.y == target_y:
                return moveNode.w
            if moveNode.x < m - 1:
                if map_int[moveNode.y][moveNode.x + 1] == 0:
                    queue.append(down(moveNode))
            if moveNode.y > 0:
                if map_int[moveNode.y - 1][moveNode.x] == 0:
                    queue.append(left(moveNode))
            if moveNode.y < n - 1:
                if map_int[moveNode.y + 1][moveNode.x] == 0:
                    queue.append(right(moveNode))
            if moveNode.x > 0:
                if map_int[moveNode.y][moveNode.x - 1] == 0:
                    queue.append(up(moveNode))


def take_point(start_x, start_y, target_x, target_y, point_list, map_int):
    num = len(point_list)
    best_way = ''
    best_way_point = ()
    min_step = 999999999
    case_n = 0
    for case in itertools.permutations(point_list, num):
        case_n += 1
        total_way = ''
        cnt = 0
        for liter_point in case:
            cnt = cnt + 1
            step_way = ''
            if cnt == 1:
                step_way = get_way(start_x, start_y, liter_point[0], liter_point[1], map_int)
            elif 1 < cnt <= num:
                step_way = get_way(last_point[0], last_point[1], liter_point[0], liter_point[1], map_int)
            last_point = liter_point
            step_way += '#'
            total_way += step_way
            if cnt == num:
                end_way = get_way(liter_point[0], liter_point[1], target_x, target_y, map_int)
                total_way += end_way
        if len(total_way) < min_step:
            min_step = len(total_way)
            best_way = total_way
            best_way_point = case
        print(case_n)
    return best_way, best_way_point


def simulate(start_x, start_y, way, point_xy, map_int):
    plt.figure(2)
    plt.clf()
    plt.imshow(map_int, cmap="Set3")
    px, py = zip(*point_xy)
    # mngr = plt.get_current_fig_manager()  # 获取当前figure manager
    # mngr.window.wm_geometry("+900+200")  # 调整窗口在屏幕上弹出的位置
    plt.scatter(px, py, marker='+', color='coral')
    robot = [start_x, start_y]
    cnt = 0
    for i in range(len(way)):
        last_robot = robot.copy()
        step = way[i]
        if step == 'L':
            robot[0] = robot[0] - 1
        elif step == 'R':
            robot[0] = robot[0] + 1
        elif step == 'U':
            robot[1] = robot[1] - 1
        elif step == 'D':
            robot[1] = robot[1] + 1
        elif step == '#':
            cnt = cnt + 1
            plt.text(robot[0], robot[1], str(cnt), fontsize=10)
            pass
        draw = [last_robot, robot]
        dx, dy = zip(*draw)
        plt.plot(dx, dy, color='blue')
    plt.pause(0.1)


# sta_x = 20
# sta_y = 1
# tar_x = 0
# tar_y = 19
# map_i = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
#          [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
#          [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
#          [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
#          [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
#          [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
#          [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#          [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
#          [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
#          [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
#          [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
#          [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
#          [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#          [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
#          [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
#
# # po_list = [[1, 5], [3, 9], [11, 7], [15, 5], [19, 15], [17, 11], [9, 13], [5, 15]]
# po_list_t = [[1, 5], [3, 9], [11, 7], [15, 5], [19, 15], [17, 11]]
#
# for point in po_list_t:
#     if map_i[point[1]][point[0]] == 1:
#         print("find wrong:")
#         print(point)
# best_w, best_p = take_point(sta_x, sta_y, tar_x, tar_y, po_list_t, map_i)
# print(best_w)
# print(len(best_w) - len(po_list_t))
# print(best_p)
# T2 = time.time()
# print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
# simulate(sta_x, sta_y, best_w, po_list_t, map_i)
