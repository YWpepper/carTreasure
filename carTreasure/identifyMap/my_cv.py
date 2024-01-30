# -*- coding:utf-8 -*-
import cv2
import numpy as np
import math


class ShapeDetector:
    def __init__(self):
        pass

    # 轮廓形状识别器 只有一个参数 c：轮廓
    # 为了进行形状检测，我们将使用轮廓近似法。 顾名思义，轮廓近似（contour approximation）是一种算法，用于通过减少一组点来减少曲线中的点数，因此称为术语近似。
    # 轮廓近似是基于以下假设：一条曲线可以由一系列短线段近似。这将导致生成近似曲线，该曲线由原始曲线定义的点子集组成。
    # 轮廓近似实际上已经通过cv2.approxPolyDP在OpenCV中实现。
    def detect(self, c):
        # 初始化形状名称，使用轮廓近似法
        # 计算周长
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # 轮廓是由一系列顶点组成的；如果是三角形，将拥有3个向量
        if len(approx) == 3:
            shape = "triangle"
        # 如果有4个顶点，那么是矩形或者正方形
        elif len(approx) == 4:
            # 计算轮廓的边界框 并且计算宽高比
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # 正方形的宽高比~~1 ，否则是矩形
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        # 否则，根据上边的膨胀腐蚀，我们假设它为圆形
        else:
            shape = "circle"
        # 返回形状的名称
        return shape


def get_four_cor(img):
    sd = ShapeDetector()
    # 转换为灰度图 高斯平滑减少高频噪音 二值化图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
    # cv2.imshow("thresh", thresh)
    # 在阈值图像上进行形状检测，并初始化形状检测器
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 注意图像是如何二值化的-形状在黑色背景上显示为白色前景。
    # 循环遍历每个轮廓，对每个轮廓执行形状检测，然后在对象上绘制形状的名称。
    detect_rect = []
    for contour_i in range(len(contours)):
        # 计算轮廓的中心，应用轮廓检测其形状
        shape = sd.detect(contours[contour_i])
        if shape == "rectangle" or shape == "square":
            # cv2.drawContours(img, [contours[contour_i]], -1, (255, 255, 0), 2)
            # cv2.imshow("contour", img)
            # print(hierarchy[0][contour_i])
            # cv2.waitKey(0)
            if hierarchy[0][contour_i][2] != -1 and hierarchy[0][contour_i][3] == 0:
                detect_rect.append(contours[contour_i])
    if len(detect_rect) != 5:
        cv2.imshow("Image", img)
        return 0
    S_list = [cv2.contourArea(i) for i in detect_rect]
    max_idx = S_list.index(max(S_list))
    detect_xy = []
    for i in range(len(detect_rect)):
        if i != max_idx:
            M = cv2.moments(detect_rect[i])
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))
            cv2.drawContours(img, [detect_rect[i]], -1, (0, 255, 0), 2)
            # cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # 展示输出图片
            detect_xy.append([cX, cY])
        else:
            cv2.drawContours(img, [detect_rect[i]], -1, (255, 255, 0), 2)

    cv2.imshow("Image", img)
    return detect_xy


def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype="float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect


def get_inner_col(img):
    sd = ShapeDetector()
    # 转换为灰度图 高斯平滑减少高频噪音 二值化图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)
    k1 = np.ones((80, 80), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k1)
    # cv2.imshow("opening", opening)
    # cv2.waitKey(0)
    k2 = np.ones((80, 80), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, k2)
    # cv2.imshow("closing", closing)
    # cv2.waitKey(0)
    inner_id = 0
    # 在阈值图像上进行形状检测，并初始化形状检测器
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓
    for k in range(len(contours)):
        shape = sd.detect(contours[k])
        if shape == "rectangle" or shape == "square":
            # debug = cv2.drawContours(img, contours[k], -1, (150, 0, 150), 4)
            # cv2.imshow("debug", debug)
            # print(hierarchy[0][k])
            # cv2.waitKey(0)
            if hierarchy[0][k][0] == -1 and hierarchy[0][k][1] == -1:
                inner_id = k

    rect = order_points(contours[inner_id].reshape(contours[inner_id].shape[0], 2))
    xs = [i[0] for i in rect]
    ys = [i[1] for i in rect]
    xs.sort()
    ys.sort()
    # 内接矩形的坐标为
    map_inner_cor1 = (int(xs[1]), int(ys[1]))
    map_inner_cor2 = (int(xs[2]), int(ys[2]))
    cv2.rectangle(img, map_inner_cor1, map_inner_cor2, (150, 0, 150), thickness=3)
    cv2.imshow("done", img)
    # map_inner_list = [[int(xs[1]), int(ys[1])], [int(xs[3]), int(ys[0])], [int(xs[0]), int(ys[3])],
    #                   [int(xs[2]), int(ys[2])]]
    # print(map_inner_list)
    img = img[map_inner_cor1[1]:map_inner_cor2[1], map_inner_cor1[0]:map_inner_cor2[0]]
    x, y = img.shape[0:2]
    max_d = max(int(x), int(y))
    img = cv2.resize(img, (max_d, max_d))
    return img


def get_circle_cor(img):
    size = img.shape
    # 转换为灰度图 高斯平滑减少高频噪音 二值化图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst_img = cv2.medianBlur(gray, 1)
    # cv2.imwrite("figure.jpg", dst_img)
    # cv2.imshow("dst_img", dst_img)

    kernel = np.ones((2, 2), dtype=np.uint8)
    erode_img = cv2.dilate(dst_img, kernel)

    # cv2.imshow("erode_img", erode_img)
    # cv2.waitKey(0)
    # 霍夫圆检测
    circle = cv2.HoughCircles(erode_img, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=14, minRadius=5, maxRadius=15)
    print(circle)
    # 将检测结果绘制在图像上
    point_list = []
    if circle is not None:
        for i in circle[0, :]:  # 遍历矩阵的每一行的数据

            [num_x, num_y] = point_map(size[0], size[1], int(i[0]), int(i[1]))
            # 绘制圆形
            cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)
            # 绘制圆心
            cv2.circle(img, (int(i[0]), int(i[1])), 2, (255, 0, 0), -1)
            cv2.putText(img, str(num_x) + "," + str(num_y), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1)
            # print([num_x, num_y])
            point_list.append([num_x, num_y])
            cv2.imshow("circle", img)
    return point_list


def point_map(w, h, x, y):
    step_x = h / 10
    step_y = w / 10
    num_x = 0
    num_y = 0
    for i in range(10):
        if i * step_x < x <= (i + 1) * step_x:
            num_x = 2 * i + 1
        else:
            pass
    for j in range(10):
        if j * step_y < y <= (j + 1) * step_y:
            num_y = 2 * j + 1
        else:
            pass
    return [num_x, num_y]


# 获得轮廓中四个角点的顺序
def range_four_col(xy4):
    x_avg = sum([point[0] for point in xy4]) / 4
    y_avg = sum([point[1] for point in xy4]) / 4
    # 左上、右上、左下、右下
    right_xy = [[0, 0], [0, 0], [0, 0], [0, 0]]
    for xy in xy4:
        if xy[0] < x_avg and xy[1] < y_avg:
            right_xy[0] = xy
        elif xy[0] > x_avg and xy[1] < y_avg:
            right_xy[1] = xy
        elif xy[0] < x_avg and xy[1] > y_avg:
            right_xy[2] = xy
        else:
            right_xy[3] = xy
    return right_xy


# 计算透视变换参数矩阵
def cal_perspective_params(img, points):
    # 设置偏移点。如果设置为(0,0),表示透视结果只显示变换的部分（也就是画框的部分）
    offset_x = 0
    offset_y = 0
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(points)
    # 透视变换的四个点
    dst = np.float32([[offset_x, offset_y], [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y], [img_size[0] - offset_x, img_size[1] - offset_y]])
    # 透视矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    print(M)
    # 透视逆矩阵
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    # print(M_inverse)
    return M, M_inverse


# 透视变换
def img_perspect_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)


def draw_line(img, points):
    # 画线
    img = cv2.line(img, points[0], points[1], (0, 0, 255), 3)
    img = cv2.line(img, points[1], points[3], (0, 0, 255), 3)
    img = cv2.line(img, points[3], points[2], (0, 0, 255), 3)
    img = cv2.line(img, points[2], points[0], (0, 0, 255), 3)
    return img
