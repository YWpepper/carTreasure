import my_cv
import time
import cv2
from SA import sa
#import serial
import numpy as np
import matplotlib.pyplot as plt

# ------------------delete repeated char -------------------#
def remove_consecutive_duplicates(input_string):
    result = input_string[0]
    for char in input_string[1:]:
        if char != result[-1]:
            result += char
    return result




def chooseWay(myChoose):
    if myChoose == 1 :      # RED
        sta_x = 20
        sta_y = 1
        tar_x = 0
        tar_y = 19
    elif myChoose==2 :       # BLUE         

        sta_x = 0
        sta_y = 19
        tar_x = 20
        tar_y = 1
        
    return  sta_x,sta_y,tar_x,tar_y


# 将输出的路径RRUULLL 进一步优化为 纯方向指令
def dealWithRoute(routeArr,baoArr,specialPoint):
    
        global result 
        for i in range(1,len(routeArr)):
            x_diff = routeArr[i][0] -  routeArr[i-1][0]
            y_diff = routeArr[i][1] -  routeArr[i-1][1]
            
           
            if x_diff > 0 :
                #print("R",end="")
                result += "R"
               
            elif x_diff < 0 :
                #print("L",end="")
                result += "L"
             
            elif y_diff > 0 :
                #print("D",end="")
                result +="D"
              
            elif y_diff < 0 :
                #print("U",end="")
                result +="U"
               
            for z in range(0,len(specialPoint)):   # 只要该坐标是宝藏坐标，则输出宝藏坐标
                if routeArr[i][0] == specialPoint[z][0] and  routeArr[i][1] == specialPoint[z][1] :
                    #print("A",end="")
                    result += "A"
            
            for j in range(0,len(baoArr)):   # 只要该坐标是宝藏坐标，则输出宝藏坐标
                if routeArr[i][0] == baoArr[j][0] and  routeArr[i][1] == baoArr[j][1] :
                    #print("#",end="")
                    result += "#"
                 
        










# ——————————————————————————————————————镜头修正函数——————————————————————————————————————
def undistort(Frame):
    fx = 469.661045287806
    cx = 314.9348873078631
    fy = 470.04020805530286
    cy = 262.6958372196564
    k1, k2, p1, p2, k3 = -0.3970199011336049, 0.15556505640757545, -0.0010010243380322113, 0.00019479138742734106, -0.031021065963771827

    # 相机坐标系到像素坐标系的转换矩阵
    k = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    # 畸变系数
    d = np.array([
        k1, k2, p1, p2, k3
    ])
    h, w = Frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(Frame, mapx, mapy, cv2.INTER_LINEAR)


def print_point(start_x, start_y, way, map_int, point_xy):
    plt.figure(2)
    plt.clf()
    plt.imshow(map_int, cmap="Set3")
    px, py = zip(*point_xy)
    plt.scatter(px, py, marker='+', color='coral')
    robot_track = []
    robot = [start_x, start_y]
    robot_track.append(robot.copy())
    cnt = 0
    for _i in range(len(way)):
        last_robot = robot.copy()
        step = way[_i]
        if step == 'L':
            robot[0] = robot[0] - 1
            robot_track.append(robot.copy())
        elif step == 'R':
            robot[0] = robot[0] + 1
            robot_track.append(robot.copy())
        elif step == 'U':
            robot[1] = robot[1] - 1
            robot_track.append(robot.copy())
        elif step == 'D':
            robot[1] = robot[1] + 1
            robot_track.append(robot.copy())
        elif step == '#':
            cnt = cnt + 1
            plt.text(robot[0], robot[1], str(cnt), fontsize=10)
            pass
        draw = [last_robot, robot]
        dx, dy = zip(*draw)
        plt.plot(dx, dy, color='blue')
    plt.show(block=False)
    return robot_track


if __name__ == '__main__':
  
    myChoose = 1  # 1 ： 红队  2 ： 蓝队 
    # 根据自己抽到的序号来确定自己是蓝队和红队
    sta_x,sta_y,tar_x,tar_y = chooseWay(myChoose)   # RED = 1  blue = 2  

    # 建立 【 存储有3条路的岔口的位置】的数组b
    specialPoint =  [[1, 11], [3, 1], [5, 5], [5, 11], [5, 13], [7, 1], [7, 7], [7, 9], [7, 11], [7, 15], [9, 3], [9, 17], [11, 3], [11, 17], [13, 5], [13, 9], [13, 11], [13, 13], [13, 19], [15, 7], [15, 9], [15, 15], [17, 19], [19, 9]]

    # ******************************************************************************************
    map_i = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
             [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
             [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
             [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
             [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
             [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
             [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
             [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
             [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
             [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    # ****************************************此处按照自己摄像头配置更改*****************************
    cap1 = cv2.VideoCapture(0)
    # ******************************************************************************************
    # 分辨率不要修改
    width = 640
    height = 480

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # ser = serial.Serial("/dev/ttyS0", 9600, timeout=0.1)

    real_point_list = []
    track_point_list = []
    cmd = 0
    print("track_cap1 is open ？ {}".format(cap1.isOpened()))

    while True:
        # 成功识别后按下回车
        # ——————————————————————————————————————路径规划流程——————————————————————————————————————
        if cmd == 13 and real_point_list:
            T1 = time.time()
            # ****************************************此处调节参数****************************************
            # 影响速度: t0, t_final, alpha, inner_iter
            all_Path, all_Ex, best_Way = sa(30, pow(10, 0), 0.9, 50, len(real_point_list), real_point_list, map_i,
                                            sta_x,
                                            sta_y,
                                            tar_x, tar_y)
            # ******************************************************************************************
            T2 = time.time()
            print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))  # 打印程序运行时间
            print("方向指令:")
            print(best_Way)  # 打印每一步的方向

            track_list = []
            for i in all_Path[-1]:
                track_list.append(real_point_list[i])
            print("宝藏点的顺序:")
            print(track_list)  # 打印宝藏点的顺序

            track = print_point(sta_x, sta_y, best_Way, map_i, real_point_list)
            print("每一步经过坐标点:")
            print(track)  # 打印每一步经过坐标点
 

            print("-----myTest--------------\n")
            result = "" 
    
            dealWithRoute(track,track_list,specialPoint)
            result = remove_consecutive_duplicates(result)
            print(result)

            print("-----myTestEnd--------------\n")
            
            
            # ser.write(best_Way)  # 此处定义想要串口发送到下位机的信息
            print("over")

            if cmd == ord('b') :
                break
            while True:
                pass
          
        # ——————————————————————————————————————图像处理流程——————————————————————————————————————
        ret, _img = cap1.read()
        # _img = undistort(_img)  # 镜头畸变校正，可以视情况使用
        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame (stream)")
            continue
        # _img = cv2.imread('book1.png')
        # 得到四个方框点和四个地图点
        raw_point = my_cv.get_four_cor(_img)
        if raw_point == 0:
            print("Can't receive picture")
            cmd = cv2.waitKey(5)
            continue
        # 将四个方框点按 左上、右上、左下、右下 排序
        right_point = my_cv.range_four_col(raw_point)
        # 根据方框点计算透视矩阵
        _M, _M_inverse = my_cv.cal_perspective_params(_img, right_point)
        # 透视转换
        tras_img = my_cv.img_perspect_transform(_img, _M)
        # cv2.imshow('tras_img', tras_img)
        # cv2.waitKey(0)
        # 根据变换后的图像寻找最大轮廓
        inner_img = my_cv.get_inner_col(tras_img)

        # cv2.imshow('inner_img', inner_img)
        # cv2.waitKey(0)
        # 寻找霍夫圆
        get_point_list = my_cv.get_circle_cor(inner_img)
        if len(get_point_list) != 8:
            print("cannot find all circle")
            cmd = cv2.waitKey(5)
            continue
        for point in get_point_list:
            if map_i[point[1]][point[0]] == 1:
                print("find wrong:")
                print(point)
                continue
        real_point_list = get_point_list
        cmd = cv2.waitKey(5)
        if cmd == ord('b') :
            break
        
