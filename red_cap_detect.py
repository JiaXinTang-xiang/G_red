# -*- coding: utf-8 -*-
"""
红色矿泉水瓶盖识别系统 - 基于HSV颜色空间
Red Bottle Cap Detection System - HSV Color Space Based

功能: 识别竖放农夫山泉等红色瓶盖，抗光照干扰
"""

import cv2
import numpy as np
import time
import serial

# =============================================================================
# 图像处理参数配置 (Image Processing Parameter Configuration)
# =============================================================================

# --- HSV 红色阈值 (HSV Red Thresholds) ---
# 红色在色相环两端，需要双范围合并
# 实测瓶盖数据: H:1-7 S:211-255 V:21-41, 均值 H=3 S=241 V=29
LOWER_RED1 = np.array([0, 145, 114])     # H:0-17, S:80-255, V:15-255
UPPER_RED1 = np.array([28, 255, 255])

LOWER_RED2 = np.array([167, 145, 114])   # H:156-180, S:80-255, V:15-255
UPPER_RED2 = np.array([180, 255, 255])

# --- 形态学参数 (Morphological Parameters) ---
MORPH_CLOSE_KERNEL = 5      # 闭运算核大小，连接断裂区域
MORPH_OPEN_KERNEL = 3       # 开运算核大小，去除小噪点
GAUSSIAN_BLUR_SIZE = 5      # 高斯模糊核大小，奇数

# --- 轮廓过滤参数 (Contour Filtering Parameters) ---
MIN_CONTOUR_AREA = 300      # 最小轮廓面积(像素²)，过滤小噪点
MAX_CONTOUR_AREA = 80000    # 最大轮廓面积(像素²)，过滤大面积背景
MIN_CIRCULARITY = 0.60      # 最小圆形度 (4π×Area/Perimeter²)，0-1之间

# --- 形态学矩形核参数 (Rectangular Kernel for Cap Shape) ---
MORPH_RECT_W = 3            # 矩形核宽度
MORPH_RECT_H = 3            # 矩形核高度

# =============================================================================
# 调试可视化开关 (Debug Visualization Switches)
# =============================================================================
DEBUG_SHOW_MASK = False          # 显示红色掩膜
DEBUG_SHOW_MORPH = False         # 显示形态学处理结果
DEBUG_SHOW_ALL = False           # 显示所有调试窗口

# =============================================================================
# 窗口管理系统 (Window Management System)
# =============================================================================
windows_created = {
    'mask': False,
    'morph': False,
}


def safe_show_window(window_name, image):
    global windows_created
    cv2.imshow(window_name, image)
    windows_created[window_name] = True


def safe_destroy_window(window_name):
    global windows_created
    if windows_created.get(window_name, False):
        cv2.destroyWindow(window_name)
        windows_created[window_name] = False


def cleanup_all_debug_windows():
    global windows_created
    for name in list(windows_created.keys()):
        safe_destroy_window(name)


# =============================================================================
# 串口通信 (UART Serial Communication)
# =============================================================================
SERIAL_PORT = '/dev/ttyS0'      # 串口号
SERIAL_BAUDRATE = 115200        # 波特率
SEND_INTERVAL = 0.1             # 发送间隔(秒)


def open_serial():
    """打开并初始化串口"""
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=SERIAL_BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1
        )
        if ser.is_open:
            print(f"串口打开成功: {ser.port}")
            ser.flushInput()
            ser.flushOutput()
        return ser
    except Exception as e:
        print(f"串口初始化失败: {e}")
        return None


def send_center_and_area_via_uart(ser, center_point, area_value):
    """
    通过UART发送中心点和面积数据
    数据包格式: [0xAA][0x06][xH][xL][yH][yL][aH][aL][0x55]
    """
    if ser is None or not ser.is_open:
        return

    x, y = map(int, center_point)
    area = int(area_value)

    # 确保值在16位范围内
    x = max(0, min(65535, x))
    y = max(0, min(65535, y))
    area = max(0, min(65535, area))

    # 构建数据包
    data_packet = bytearray()
    data_packet.append(0xAA)  # 帧头
    data_packet.append(0x06)  # 数据长度

    # X坐标 (高位在前)
    data_packet.append((x >> 8) & 0xFF)  # xH
    data_packet.append(x & 0xFF)         # xL

    # Y坐标
    data_packet.append((y >> 8) & 0xFF)  # yH
    data_packet.append(y & 0xFF)         # yL

    # 面积
    data_packet.append((area >> 8) & 0xFF)  # aH
    data_packet.append(area & 0xFF)         # aL

    data_packet.append(0x55)  # 帧尾

    ser.write(data_packet)


# =============================================================================
# HSV交互分析 (Interactive HSV Analysis)
# =============================================================================
current_frame_for_hsv = None
clicked_point = None
hsv_output_requested = False


def mouse_callback(event, x, y, flags, param):
    global clicked_point, hsv_output_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        hsv_output_requested = True


def output_hsv_info(image, x, y, region_size=5):
    """分析点击位置的HSV信息，辅助阈值标定"""
    try:
        h, w = image.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ph, ps, pv = hsv[y, x]
        print(f"\n=== 点击位置({x},{y}) HSV: H={ph} S={ps} V={pv} ===")

        half = region_size // 2
        y1, y2 = max(0, y - half), min(h, y + half + 1)
        x1, x2 = max(0, x - half), min(w, x + half + 1)
        region = hsv[y1:y2, x1:x2]

        h_min, s_min, v_min = np.min(region, axis=(0, 1))
        h_max, s_max, v_max = np.max(region, axis=(0, 1))
        h_mean, s_mean, v_mean = np.mean(region, axis=(0, 1)).astype(int)

        print(f"区域HSV范围({region_size}x{region_size}):")
        print(f"  H: {h_min}-{h_max} (均值:{h_mean})")
        print(f"  S: {s_min}-{s_max} (均值:{s_mean})")
        print(f"  V: {v_min}-{v_max} (均值:{v_mean})")

        # 建议阈值
        h_tol = max(10, (h_max - h_min) + 5)
        s_tol = max(40, (s_max - s_min) + 20)
        v_tol = max(40, (v_max - v_min) + 20)

        if h_mean < 15 or h_mean > 155:
            print("\n检测到红色，建议双范围阈值:")
            if h_mean < 15:
                print(f"  lower_red1 = np.array([0, {max(0, s_min-20)}, {max(0, v_min-20)}])")
                print(f"  upper_red1 = np.array([{min(180, h_max+10)}, 255, 255])")
                print(f"  lower_red2 = np.array([156, {max(0, s_min-20)}, {max(0, v_min-20)}])")
                print(f"  upper_red2 = np.array([180, 255, 255])")
            else:
                print(f"  lower_red1 = np.array([0, {max(0, s_min-20)}, {max(0, v_min-20)}])")
                print(f"  upper_red1 = np.array([{min(180, h_max-160+10)}, 255, 255])")
                print(f"  lower_red2 = np.array([{max(0, h_min-10)}, {max(0, s_min-20)}, {max(0, v_min-20)}])")
                print(f"  upper_red2 = np.array([180, 255, 255])")
        print("=" * 50)
    except Exception as e:
        print(f"HSV分析失败: {e}")


# =============================================================================
# 红色瓶盖检测核心算法 (Red Cap Detection Core Algorithm)
# =============================================================================

def detect_red_cap(frame):
    """
    红色瓶盖检测主函数
    返回: (result_frame, cap_center, cap_radius, mask_red)
        cap_center: (x, y) 圆心坐标，未检测到时为 None
        cap_radius: 半径，未检测到时为 0
    """
    # 步骤1: 高斯模糊降噪
    blurred = cv2.GaussianBlur(frame, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)

    # 步骤2: 转换到HSV颜色空间
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 步骤3: 双范围红色掩膜
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # 步骤4: 形态学处理 - 闭运算连接断开的红色区域
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))
    mask_closed = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_close)

    # 步骤5: 形态学开运算 - 去除小噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL))
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

    # 步骤6: 查找轮廓
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cap_center = None
    cap_radius = 0
    cap_contour = None

    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 面积过滤
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        # 圆形度过滤
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < MIN_CIRCULARITY:
            continue

        # 最小外接圆拟合
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)

        # 综合评分：圆形度越高越好，面积适中越好
        area_score = min(area / 5000.0, 5000.0 / max(area, 1))
        score = circularity * 0.7 + area_score * 0.3

        if score > best_score:
            best_score = score
            cap_center = (int(cx), int(cy))
            cap_radius = int(radius)
            cap_contour = cnt

    return mask_clean, cap_center, cap_radius, cap_contour


def draw_detection_result(frame, cap_center, cap_radius, cap_contour, mask_clean):
    """在图像上绘制检测结果"""
    if cap_center is not None and cap_radius > 0:
        # 绘制红色掩膜轮廓
        if cap_contour is not None:
            cv2.drawContours(frame, [cap_contour], -1, (0, 255, 0), 2)

        # 绘制圆心和最小外接圆
        cv2.circle(frame, cap_center, 5, (0, 255, 255), -1)
        cv2.circle(frame, cap_center, cap_radius, (255, 0, 0), 2)

        # 绘制十字线
        cross_size = max(10, cap_radius // 2)
        cv2.line(frame,
                 (cap_center[0] - cross_size, cap_center[1]),
                 (cap_center[0] + cross_size, cap_center[1]),
                 (0, 255, 255), 1)
        cv2.line(frame,
                 (cap_center[0], cap_center[1] - cross_size),
                 (cap_center[0], cap_center[1] + cross_size),
                 (0, 255, 255), 1)

        # 标注圆心和半径
        label = f"Center:({cap_center[0]},{cap_center[1]}) R:{cap_radius}"
        cv2.putText(frame, label, (cap_center[0] + 15, cap_center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if cap_contour is not None:
            area = cv2.contourArea(cap_contour)
            cv2.putText(frame, f"Area:{area:.0f}", (cap_center[0] + 15, cap_center[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return True
    return False


# =============================================================================
# 主运行函数 (Main Run Function)
# =============================================================================

def run():
    global clicked_point, hsv_output_requested, current_frame_for_hsv
    global DEBUG_SHOW_MASK, DEBUG_SHOW_MORPH, DEBUG_SHOW_ALL
    global LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2
    global MIN_CONTOUR_AREA, MIN_CIRCULARITY, MORPH_CLOSE_KERNEL, MORPH_OPEN_KERNEL

    print("=" * 60)
    print("  红色矿泉水瓶盖识别系统 - Red Cap Detection")
    print("=" * 60)

    # 打开摄像头
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    if not cap.isOpened():
        # 尝试备用索引
        cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    if not cap.isOpened():
        # 最后尝试默认后端
        cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("无法打开摄像头，尝试读取测试图片模式...")
        cap = None

    if cap is not None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)

    # 初始化串口
    ser = open_serial()

    cv2.namedWindow("Red Cap Detection", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Red Cap Detection", mouse_callback)

    fps = 0
    frame_count = 0
    t_start = time.time()

    # 串口发送控制
    last_center = None
    last_area = None
    last_send_time = 0

    # 加载测试图片（摄像头不可用时）
    test_frame = None
    if cap is None:
        test_path = "test_bottle.jpg"
        if not cv2.haveImageReader(test_path):
            print(f"请将测试图片放到当前目录: {test_path}")
            print("或连接摄像头后重新运行")
            cv2.destroyAllWindows()
            return
        test_frame = cv2.imread(test_path)
        if test_frame is None:
            print("读取测试图片失败")
            cv2.destroyAllWindows()
            return
        print(f"使用测试图片: {test_path}")

    print("\n操作说明:")
    print("  Q/ESC - 退出")
    print("  S    - 保存当前帧")
    print("  M    - 切换红色掩膜显示")
    print("  P    - 切换形态学结果显示")
    print("  D    - 切换所有调试显示")
    print("  R    - 重置点击点")
    print("  + / - - 调整最小轮廓面积阈值")
    print("  [ / ] - 调整圆形度阈值")
    print("  ; / ' - 调整闭运算核大小")
    print("  , / . - 调整开运算核大小")
    print("  1    - 打印当前所有参数")
    print("  鼠标点击 - 获取该点HSV颜色信息")
    print("=" * 60)

    while True:
        frame_start = time.time()

        # --- 图像采集 ---
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("读取帧失败")
                break
        else:
            frame = test_frame.copy()

        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        current_frame_for_hsv = frame.copy()

        # --- 红色瓶盖检测 ---
        mask_clean, cap_center, cap_radius, cap_contour = detect_red_cap(frame)

        # --- 绘制检测结果 ---
        result = frame.copy()
        detected = draw_detection_result(result, cap_center, cap_radius, cap_contour, mask_clean)

        # --- 串口发送坐标和面积 ---
        current_time = time.time()
        if detected and cap_contour is not None:
            area = cv2.contourArea(cap_contour)
            last_center = cap_center
            last_area = area
            if current_time - last_send_time >= SEND_INTERVAL:
                send_center_and_area_via_uart(ser, cap_center, area)
                last_send_time = current_time
        elif last_center is not None and last_area is not None:
            # 当前帧未检测到，使用上一帧有效数据继续发送
            if current_time - last_send_time >= SEND_INTERVAL:
                send_center_and_area_via_uart(ser, last_center, last_area)
                last_send_time = current_time

        # --- 绘制点击点 ---
        if clicked_point is not None:
            cv2.circle(result, clicked_point, 5, (0, 0, 255), -1)
            cv2.putText(result, f"Click:({clicked_point[0]},{clicked_point[1]})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if hsv_output_requested:
                output_hsv_info(current_frame_for_hsv, clicked_point[0], clicked_point[1])
                hsv_output_requested = False

        # --- FPS计算 ---
        frame_count += 1
        elapsed = time.time() - t_start
        if elapsed > 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            t_start = time.time()

        frame_time = (time.time() - frame_start) * 1000

        # --- 状态信息显示 ---
        status = "DETECTED" if detected else "SEARCHING..."

        cv2.putText(result, f"Status: {status}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if detected else (0, 0, 255), 2)
        cv2.putText(result, f"FPS: {fps:.1f} | {frame_time:.0f}ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        uart_status = "UART:OK" if (ser is not None and ser.is_open) else "UART:OFF"
        uart_color = (0, 255, 0) if (ser is not None and ser.is_open) else (0, 0, 255)
        cv2.putText(result, uart_status, (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, uart_color, 1)

        # --- 底部提示 ---
        hint_y = result.shape[0] - 55
        cv2.putText(result,
                    "Q:Quit S:Save M:Mask P:Morph D:Debug 1:Params +/-:Area []:Circ",
                    (5, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(result,
                    f"Area>{MIN_CONTOUR_AREA} Circ>{MIN_CIRCULARITY:.2f} CloseK{MORPH_CLOSE_KERNEL} OpenK{MORPH_OPEN_KERNEL}",
                    (5, hint_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(result,
                    f"Red1:[0-{UPPER_RED1[0]},S>{LOWER_RED1[1]},V>{LOWER_RED1[2]}] Red2:[{LOWER_RED2[0]}-180,S>{LOWER_RED2[1]},V>{LOWER_RED2[2]}]",
                    (5, hint_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)
        cv2.putText(result,
                    f"Blue Cap Threshold: S>{LOWER_RED1[1]} V>{LOWER_RED1[2]} | Click for HSV",
                    (5, hint_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 200, 0), 1)

        cv2.imshow("Red Cap Detection", result)

        # --- 调试窗口 ---
        if DEBUG_SHOW_MASK or DEBUG_SHOW_ALL:
            mask_color = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
            safe_show_window("mask", mask_color)
        else:
            safe_destroy_window("mask")

        if DEBUG_SHOW_MORPH or DEBUG_SHOW_ALL:
            morph_display = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
            if cap_contour is not None:
                cv2.drawContours(morph_display, [cap_contour], -1, (0, 255, 0), 2)
            safe_show_window("morph", morph_display)
        else:
            safe_destroy_window("morph")

        # --- 键盘处理 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q 或 ESC
            print("用户退出")
            break

        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"cap_detect_{ts}.jpg", result)
            print(f"已保存: cap_detect_{ts}.jpg")

        elif key == ord('r'):
            clicked_point = None
            print("已重置点击点")

        elif key == ord('m'):
            DEBUG_SHOW_MASK = not DEBUG_SHOW_MASK
            print(f"红色掩膜显示: {'开' if DEBUG_SHOW_MASK else '关'}")

        elif key == ord('p'):
            DEBUG_SHOW_MORPH = not DEBUG_SHOW_MORPH
            print(f"形态学结果显示: {'开' if DEBUG_SHOW_MORPH else '关'}")

        elif key == ord('d'):
            DEBUG_SHOW_ALL = not DEBUG_SHOW_ALL
            DEBUG_SHOW_MASK = DEBUG_SHOW_ALL
            DEBUG_SHOW_MORPH = DEBUG_SHOW_ALL
            print(f"所有调试显示: {'开' if DEBUG_SHOW_ALL else '关'}")

        elif key == ord('+') or key == ord('='):
            MIN_CONTOUR_AREA = min(50000, MIN_CONTOUR_AREA + 100)
            print(f"最小轮廓面积: {MIN_CONTOUR_AREA}")

        elif key == ord('-'):
            MIN_CONTOUR_AREA = max(50, MIN_CONTOUR_AREA - 100)
            print(f"最小轮廓面积: {MIN_CONTOUR_AREA}")

        elif key == ord('['):
            MIN_CIRCULARITY = max(0.1, MIN_CIRCULARITY - 0.05)
            print(f"最小圆形度: {MIN_CIRCULARITY:.2f}")

        elif key == ord(']'):
            MIN_CIRCULARITY = min(1.0, MIN_CIRCULARITY + 0.05)
            print(f"最小圆形度: {MIN_CIRCULARITY:.2f}")

        elif key == ord(';'):
            MORPH_CLOSE_KERNEL = max(1, MORPH_CLOSE_KERNEL - 2)
            print(f"闭运算核大小: {MORPH_CLOSE_KERNEL}")

        elif key == ord('\''):
            MORPH_CLOSE_KERNEL = min(31, MORPH_CLOSE_KERNEL + 2)
            print(f"闭运算核大小: {MORPH_CLOSE_KERNEL}")

        elif key == ord(','):
            MORPH_OPEN_KERNEL = max(1, MORPH_OPEN_KERNEL - 2)
            print(f"开运算核大小: {MORPH_OPEN_KERNEL}")

        elif key == ord('.'):
            MORPH_OPEN_KERNEL = min(31, MORPH_OPEN_KERNEL + 2)
            print(f"开运算核大小: {MORPH_OPEN_KERNEL}")

        elif key == ord('1'):
            print("\n========== 当前参数 ==========")
            print(f"HSV Red1: H[0-{UPPER_RED1[0]}] S[{LOWER_RED1[1]}-255] V[{LOWER_RED1[2]}-255]")
            print(f"HSV Red2: H[{LOWER_RED2[0]}-180] S[{LOWER_RED2[1]}-255] V[{LOWER_RED2[2]}-255]")
            print(f"高斯模糊核: {GAUSSIAN_BLUR_SIZE}")
            print(f"闭运算核: {MORPH_CLOSE_KERNEL}  开运算核: {MORPH_OPEN_KERNEL}")
            print(f"轮廓面积: {MIN_CONTOUR_AREA}-{MAX_CONTOUR_AREA}")
            print(f"最小圆形度: {MIN_CIRCULARITY:.2f}")
            print(f"调试窗口: mask={DEBUG_SHOW_MASK} morph={DEBUG_SHOW_MORPH} all={DEBUG_SHOW_ALL}")
            print("==============================")

    # 清理
    if cap is not None:
        cap.release()
    if ser is not None and ser.is_open:
        ser.close()
        print("串口已关闭")
    cleanup_all_debug_windows()
    cv2.destroyAllWindows()
    print("程序结束")


if __name__ == "__main__":
    run()
