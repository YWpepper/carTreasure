import cv2
import numpy as np
# import serial


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


# ser = serial.Serial("/dev/ttyS0", 9600, timeout=0.1)

# red1
r_low_hsv1 = np.array([160, 43, 60])
r_high_hsv1 = np.array([180, 255, 255])
# red2
r_low_hsv2 = np.array([0, 43, 60])
r_high_hsv2 = np.array([10, 255, 255])
# green
g_low_hsv1 = np.array([50, 43, 60])
g_high_hsv1 = np.array([94, 255, 255])
# blue
b_low_hsv1 = np.array([100, 43, 60])
b_high_hsv1 = np.array([124, 255, 255])
# yellow
y_low_hsv1 = np.array([11, 43, 60])
y_high_hsv1 = np.array([31, 255, 255])
font = cv2.FONT_HERSHEY_SIMPLEX

# img_src = cv2.imread("3.png")
cap1 = cv2.VideoCapture(0)

width = 1920
height = 1080

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


def image_tf_judge(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # mask是只突出指定颜色的图片
    mask_red = cv2.inRange(hsv, lowerb=r_low_hsv1, upperb=r_high_hsv1) + cv2.inRange(hsv, lowerb=r_low_hsv2,
                                                                                     upperb=r_high_hsv2)
    mask_blue = cv2.inRange(hsv, lowerb=b_low_hsv1, upperb=b_high_hsv1)
    mask_yellow = cv2.inRange(hsv, lowerb=y_low_hsv1, upperb=y_high_hsv1)
    mask_green = cv2.inRange(hsv, lowerb=g_low_hsv1, upperb=g_high_hsv1)

    inner_mask = [mask_yellow, mask_green]
    inner_mask_S = []
    inner_mask_counter = []
    outer_mask = [mask_blue + mask_yellow + mask_green, mask_red + mask_yellow + mask_green]
    outer_mask_S = []
    outer_mask_counter = []
    features = {"inner_color": "none", "inner_shape": "none", "outer_color": "none"}  # 内部颜色 内部形状 外部颜色 外部形状
    sd = ShapeDetector()
    detect = 0
    for i in range(len(inner_mask)):
        # 中值滤波降噪
        median = cv2.medianBlur(inner_mask[i], 7)
        k1 = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, k1)
        cv2.imshow("inner" + str(i), opening)
        contours, hierarchy = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            area = []
            # 找到最大的轮廓
            for k in range(len(contours)):
                area.append(cv2.contourArea(contours[k]))
            max_idx = np.argmax(np.array(area))
            inner_mask_S.append(area[max_idx])
            inner_mask_counter.append(contours[max_idx])
        else:
            inner_mask_S.append(0)
            inner_mask_counter.append(None)
    inner_tag = inner_mask_S.index(max(inner_mask_S))
    if inner_tag == 0:
        features["inner_color"] = "yellow"
    else:
        features["inner_color"] = "green"
    if inner_mask_counter[inner_tag] is not None:
        shape = sd.detect(inner_mask_counter[inner_tag])
        if shape == "triangle":
            features["inner_shape"] = "triangle"
        elif shape == "circle":
            features["inner_shape"] = "circle"

    for i in range(len(outer_mask)):
        # 中值滤波降噪
        median = cv2.medianBlur(outer_mask[i], 7)
        k1 = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, k1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, k1)
        cv2.imshow("outer" + str(i), closing)
        contours, hierarchy = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            area = []
            # 找到最大的轮廓
            for k in range(len(contours)):
                area.append(cv2.contourArea(contours[k]))
            max_idx = np.argmax(np.array(area))
            outer_mask_S.append(area[max_idx])
            outer_mask_counter.append(contours[max_idx])
        else:
            outer_mask_S.append(0)
            outer_mask_counter.append(None)
    outer_tag = outer_mask_S.index(max(outer_mask_S))
    if outer_tag == 0:
        features["outer_color"] = "blue"
    else:
        features["outer_color"] = "red"
    if inner_mask_counter[inner_tag] is not None:
        M = cv2.moments(inner_mask_counter[inner_tag])  # 求矩
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])  # 求x坐标
            cy = int(M['m01'] / M['m00'])  # 求y坐标
            x, y, w, h = cv2.boundingRect(outer_mask_counter[outer_tag])
            if x < cx < x + w and y < cy < y + w:
                print(inner_mask_S[inner_tag] / outer_mask_S[outer_tag])
                if 0.05 < (inner_mask_S[inner_tag] / outer_mask_S[outer_tag]) < 0.3:
                    if features["inner_color"] == "yellow" and features["inner_shape"] == "circle" and features[
                        "outer_color"] == "blue":
                        detect = 1
                    elif features["inner_color"] == "green" and features["inner_shape"] == "triangle" and features[
                        "outer_color"] == "red":
                        detect = 2
                    elif features["inner_color"] == "green" and features["inner_shape"] == "triangle" and features[
                        "outer_color"] == "blue":
                        detect = 3
                    elif features["inner_color"] == "yellow" and features["inner_shape"] == "circle" and features[
                        "outer_color"] == "red":
                        detect = 4
    if detect != 0:
        x, y, w, h = cv2.boundingRect(outer_mask_counter[outer_tag])
        img_boundingRect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
        cv2.putText(img_boundingRect,
                    features["inner_color"] + "_" + features["inner_shape"] + "_" + features["outer_color"],
                    (int(x + w / 2), int(y + h / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1)
        cv2.imshow("img", img_boundingRect)
        print(features["inner_color"] + "_" + features["inner_shape"] + "_" + features["outer_color"])
    if detect == 1:  # 此处改成己方的真宝藏
        print("true")
        # serial.write(detect) # 串口输出己方的真宝藏
    if detect == 2 or detect == 2 or detect == 3:  # 此处改成非己方的宝藏
        print("false")
        # serial.write(detect) # 串口输出己方的假宝藏
    else:
        cv2.imshow("img", img)


if cap1.isOpened():
    while True:
        ret, frame = cap1.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame (stream)")
        image_tf_judge(frame)
        cv2.waitKey(1)
