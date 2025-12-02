from ultralytics import YOLO
import cv2
import json

import numpy as np
import socket
import threading

import collections
import time

import os
from pynput import keyboard
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import sys

frame_cnt = 0
i=0
f=2
#1280 720
WIDTH = 640
HEIGHT = 480
#NUM = 50000
FPS = 5

# 変更: 遅延用にフレームを保存するデックを定義（3秒分のフレームを保存）
# FPS = 5 の場合、3秒分のフレームは 3秒 * 5フレーム = 15フレーム
delay_frames = 12  # ?秒遅延させるためのフレーム数
#frame_buffer1 = collections.deque(maxlen=delay_frames)  # カメラ1用バッファ
#frame_buffer2 = collections.deque(maxlen=delay_frames)  # カメラ2用バッファ
#frame_buffer = [[None for _ in range(delay_frames)] for _ in range(4)]  # カメラ2用の2次元配列
frame_buffer = [collections.deque(maxlen=delay_frames) for _ in range(4)]

#ソケット通信
ip="127.0.0.1"
port=8000

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((ip,port))
# キーが離されたときに呼び出される関数
def on_key_release(key):
    global stop_flag
    try:
        if key == keyboard.Key.esc:
            print("ESCキーが押されました。終了します。")
            s.send("esc".encode("utf-8"))  # ESCメッセージを送信
            stop_flag = True
            return False  # リスナーを停止してプログラムを終了
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        stop_flag = True
        return False

print("Connected!!!!!")
def webcamera_set(cap0):
    cap0.set(cv2.CAP_PROP_EXPOSURE,-7)
    #cap0.set(cv2.CAP_PROP_AUTOFOCUS,)
    
    cap0.set(cv2.CAP_PROP_FOCUS,240)
    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)


    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cap0.set(cv2.CAP_PROP_FPS, FPS)
    
def add_frame_to_buffer(camera_index, frame, buffer):
    # 古いフレームをシフトして、新しいフレームを最後に追加
    buffer[camera_index] = buffer[camera_index][1:] + [frame]

def detect_target(frame):
    #img = cv2.imread(img_name) # 画像を読み込む
    yo=0
    img=frame.copy()
    H1_max = 25
    H1_min = 0
    S1_max = 255
    S1_min = 70
    V1_max = 255
    V1_min = 60

    H2_max = 180
    H2_min = 165
    S2_max = 255
    S2_min = 70
    V2_max = 255
    V2_min = 60

    # 赤色は２つの領域にまたがります！！
    # np.array([色彩, 彩度, 明度])
    # 各値は適宜設定する！！
    HIGH_COLOR1 = np.array([H1_max, S1_max, V1_max]) # 各最大値を指定
    LOW_COLOR1 = np.array([H1_min, S1_min, V1_min]) # 各最小値を指定
    HIGH_COLOR2 = np.array([H2_max, S2_max, V2_max])
    LOW_COLOR2 = np.array([H2_min, S2_min, V2_min])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # BGRからHSVに変換
    # マスクを作成
    bin_img1 = cv2.inRange(hsv, LOW_COLOR1, HIGH_COLOR1) 
    bin_img2 = cv2.inRange(hsv, LOW_COLOR2, HIGH_COLOR2)
    mask = bin_img1 + bin_img2 # 必要ならマスクを足し合わせる
    masked_img = cv2.bitwise_and(img, img, mask= mask) # 元画像から特定の色を抽出
    
     # 連結成分でラベリング
    num_labels, label_img, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    out_img = masked_img.copy()
    
    coordinates = []
    
    if num_labels >= 1: # ラベルの有無で場合分け
        for label in range(1, num_labels):  # 背景ラベル(0)を無視
        # 以下最大面積のラベルについて考える
            x = stats[label][0]
            y = stats[label][1]
            w = stats[label][2]
            h = stats[label][3]
            s = stats[label][4]
            mx = int(centroids[label][0]) # 重心のX座標
            my = int(centroids[label][1]) # 重心のY座標
            if mx>100 and mx<1200:
                if s>10000:
                    yo=1
                    #cv2.rectangle(out_img, (x, y), (x+w, y+h), (0, 255, 0),3) # ラベルを四角で囲む
                    #cv2.putText(out_img, "%d,%d"%(mx, my), (x-15, y+h+15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0)) # 重心を表示
                    #cv2.putText(out_img, "%d"%(s), (x, y+h+30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0)) # 面積を表示
                    # 各領域の座標情報をリストに格納
                    coordinates.append((y, y+h, x, x+w, mx, my, s))
                    

    #result_img = fill_non_detected_with_white(img, coordinates)
    return yo,coordinates, num_labels - 1  # 背景ラベルを除く
    #cv2.imwrite("out_img.jpg", out_img) # 書き出す

#トリミングして保存
def save(i,image, coordinates,labels):

    if labels > 0:
        for (top, bottom, left, right) in coordinates:
            image_red = image[top:bottom, left:right]
            cv2.imwrite(f'image/{i:04d}.png', image_red)
            i += 1
    else:
        image_red=image.copy()
    return i, image_red

def crop_center_480x480(image):
    """
    画像の中心部分を480x480で切り抜く関数
    :param image: 元画像
    :return: 切り抜かれた480x480の画像
    """
    height, width = image.shape[:2]
    
    # 画像の中心座標を計算
    center_x, center_y = width // 2, height // 2
    
    # 切り抜き範囲の左上の座標を計算
    top_left_x = center_x - 240
    top_left_y = center_y - 240
    
    # 切り抜き範囲の右下の座標を計算
    bottom_right_x = center_x + 240
    bottom_right_y = center_y + 240
    
    # 画像の範囲を超えないようにクリップする
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(width, bottom_right_x)
    bottom_right_y = min(height, bottom_right_y)

    # 480x480の中央部分を切り抜き
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    return cropped_image
def crop_center_640x640(image):
    """
    画像の中心部分を640x640で切り抜く関数
    :param image: 元画像
    :return: 切り抜かれた640x640の画像
    """
    height, width = image.shape[:2]
    
    # 画像の中心座標を計算
    center_x, center_y = width // 2, height // 2
    
    # 切り抜き範囲の左上の座標を計算
    top_left_x = center_x - 320
    top_left_y = center_y - 320
    
    # 切り抜き範囲の右下の座標を計算
    bottom_right_x = center_x + 320
    bottom_right_y = center_y + 320
    
    # 画像の範囲を超えないようにクリップする
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(width, bottom_right_x)
    bottom_right_y = min(height, bottom_right_y)

    # 640x640の中央部分を切り抜き
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    return cropped_image


# 画像を640x640にリサイズする関数
def resize_image(frame, coordinates, labels):
    target_size=(640, 640)
    image=frame.copy()
    # デフォルトで空の画像を用意
    #image_red=np.full((640, 640, 3),0, dtype=np.uint8)
    
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    size = min(width, height) // 2
    image_red = image[center_y - size:center_y + size, center_x - size:center_x + size]

    if labels > 0:
        
        for (top, bottom, left, right, mx, my, s) in coordinates:
            if mx>100 and mx<1200:
                if s>10000:
                    h=bottom-top
                    w=right-left
                    if h<w:
                        bottom=top+w
                        
                    else:
                        right=left+h

                    image_red = image[top:bottom, left:right]
                    break
                else:
                    
                    height, width = image.shape[:2]
                    center_x, center_y = width // 2, height // 2
                    size = min(width, height) // 2
                    image_red = image[center_y - size:center_y + size, center_x - size:center_x + size]
                    break
            else:
                
                height, width = image.shape[:2]
                center_x, center_y = width // 2, height // 2
                size = min(width, height) // 2
                image_red = image[center_y - size:center_y + size, center_x - size:center_x + size]
    else:
        
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        size = min(width, height) // 2
        image_red = image[center_y - size:center_y + size, center_x - size:center_x + size]
    
    if image_red.size == 0:
        # エラー処理またはデフォルトの画像を設定
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        size = min(width, height) // 2
        image_red = image[center_y - size:center_y + size, center_x - size:center_x + size]
        print("Error: The image_red is empty. Check the coordinates or input image.")
        return image_red

    #リサイズ
    resized_image = cv2.resize(image_red, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

# 映像をリサイズし1つの画面に統合する関数
def combine_frames(frames):
    # 各フレームをリサイズ (320x320)にする
    resized_frames = [cv2.resize(frame, (320, 320)) for frame in frames]
    
    # 2x2のグリッドに配置
    top_row = np.hstack((resized_frames[0], resized_frames[1]))
    bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
    
    # 上下を縦に結合
    combined_frame = np.vstack((top_row, bottom_row))
    
    return combined_frame

def run_tracker_in_thread_d(cam1, cam2, model1, model2, output_frames, cls, index1, index2,yo):
    cap1 = cv2.VideoCapture(cam1,cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(cam2,cv2.CAP_DSHOW)
    print("cam")
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("カメラが開けません")
        print(cam1,cam2)
        return

    webcamera_set(cap1)
    webcamera_set(cap2)
    try:
        while cap1.isOpened() and cap2.isOpened():
            success1, frame1 = cap1.read()
            success2, frame2 = cap2.read()
            
            
            if success1 and success2:
                # フレームをバッファに追加
                frame_buffer[index1].append(frame1)
                #print(len(frame_buffer[index1]))

                if len(frame_buffer[index1]) < delay_frames:
                    # バッファが満たされていない場合はフレームをスキップ、もしくはデフォルト処理
                    print(f"frame_buffer[{index1}] にはまだ十分なフレームが揃っていません")
                    continue

                # 3秒遅延させるため、バッファが埋まっている場合に処理を行う
                elif len(frame_buffer[index1]) == delay_frames :
                    # フレーム数が delay_frames と等しい場合の処理
                    #print(f"frame_buffer[{index1}] は {delay_frames} フレーム揃いました")
                    # 3秒前のフレームを使用
                    delayed_frame1 = frame_buffer[index1][0]

                    yo[index1],coordinates1, labels1 = detect_target(delayed_frame1)
                    yo[index2],coordinates2, labels2 = detect_target(frame2)

                    img1 = resize_image(delayed_frame1, coordinates1, labels1)
                    img2 = resize_image(frame2, coordinates2, labels2)
                    
                    if yo[0]==1 or yo[1]==1 or yo[2]==1 or yo[3]==1:
                        results1 = model1.track(source=img1, persist=True, classes=[0, 1])
                        results2 = model2.track(source=img2, persist=True, classes=[0, 1])

                        # 推論結果を注釈（YOLO推論が空の場合でもフレームを使用）
                        if len(results1) > 0:
                            annotated_frame1 = results1[0].plot()
                        else:
                            annotated_frame1 = delayed_frame1  # YOLO結果が空なら元のフレームを使用

                        if len(results2) > 0:
                            annotated_frame2 = results2[0].plot()
                        else:
                            annotated_frame1 = delayed_frame1  # YOLO結果が空なら元のフレームを使用
                            annotated_frame2 = frame2  # YOLO結果が空なら元のフレームを使用
                        
                        #with lock:
                        output_frames[index1] = annotated_frame1
                        output_frames[index2] = annotated_frame2

                        items1 = results1[0]
                        items2 = results2[0]

                        for item1 in items1:
                            cls[index1] = int(item1.boxes.cls)
                        for item2 in items2:
                            cls[index2] = int(item2.boxes.cls)
                    else:
                        annotated_frame1 = delayed_frame1  # YOLO結果が空なら元のフレームを使用
                        annotated_frame2 = frame2  # YOLO結果が空なら元のフレームを使用

                        output_frames[index1] = annotated_frame1
                        output_frames[index2] = annotated_frame2
                        

                    

                    
                else:
                    # フレームがまだ揃っていない場合は処理をスキップするか、別の処理を行う
                    frame_buffer[index1].pop(0)  # 最も古いフレームを削除
                    print(f"2frame_buffer[{index1}] のフレームが超えました")
                    print(frame_buffer[index1][0].shape[0])
                    cv2.imshow("View", frame_buffer[index1][0])
                    continue  # この場合、ループを続けて次のフレームを処理
                # 'q'キーで終了
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    message = "q"
                    s.send(message.encode("utf-8"))
                    #receive = s.recv(4096).decode()
                    #print("Received from server:", receive)
                    break
            
    finally:
        cap1.release()
        cap2.release()
        #frame_buffer.clear()  # deque全体をクリア

# キーボードリスナーを開始 (Escキーのみ監視)
listener = keyboard.Listener(on_release=on_key_release)
listener.start()

#model = YOLO("yolov8x.pt")
model1=YOLO("C:/gamo/yolo_v1/runs/detect/train7/weights/best.pt")
model2=YOLO("C:/gamo/yolo_v1/runs/detect/train7/weights/best.pt")
model3=YOLO("C:/gamo/yolo_v1/runs/detect/train7/weights/best.pt")
model4=YOLO("C:/gamo/yolo_v1/runs/detect/train7/weights/best.pt")

# Open the video file
#video_path = "./sample/sample_1.mp4"
#video_path1 = "1_1_cam0.mov"
#video_path2 = "mold/bottom_cam1.mov"
#video_path3 = "bc/in_cam1.mov"
#video_path4 = "unripe/1_cam0.mov"

#json
with open("C:/camera_set/camera_index.json","r")as file:
    config=json.load(file)

camera_index=config.get("camera_index",[])

# Define the video files or camera sources for the trackers
video_path1 = camera_index[0]  # First camera
video_path2 = camera_index[1]   # Second camera
video_path3 = camera_index[2]   # Third camera
video_path4 = camera_index[3]   # Fourth camera

# カメラまたはビデオのフレームを格納するためのリスト
output_frames = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(4)]
cls=[100]*4
yo=[100]*4



# Create the tracker threads for 4 cameras
tracker_thread1 = threading.Thread(target=run_tracker_in_thread_d, args=(video_path1, video_path2, model1, model2, output_frames, cls, 0, 1,yo), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread_d, args=(video_path3, video_path4, model3, model4, output_frames, cls, 2, 3,yo), daemon=True)

# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()

while True:
    # フレームを1つのウィンドウに統合して表示
    combined_frame = combine_frames(output_frames)
    
    # 1つのウィンドウで表示
    cv2.imshow("Combined View", combined_frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me = "9"#被害果の出力
        s.send(message.encode("utf-8"))
        #receive = s.recv(4096).decode()
        #print("Received from server:", receive)
        print("qキーが押されました。プログラムを終了します。")
        sys.exit()
    
    try:
        #print(yo)
        #print(yo2)
        
        if yo[0]==1 or yo[1]==1 or yo[2]==1 or yo[3]==1:
            if cls[0] in [0, 1, 2] or cls[1] in [0, 1, 2] or cls[2] in [0, 1, 2] or cls[3] in [0, 1, 2]:
                if f == 2 or f==1:
                    message = "0"#被害果の出力
                    s.send(message.encode("utf-8"))
                    #receive = s.recv(4096).decode()
                    #print("Received from server:", receive)
                    print(message)
                    f = 0
            else:
                if f == 2:
                    message = "1"
                    s.send(message.encode("utf-8"))
                    print(message)
                    #receive = s.recv(4096).decode()
                    #print("Received from server:", receive)
                    f=1
        else:
            if f == 1 or f==0:
                message = "2"
                s.send(message.encode("utf-8"))
                print(message)
                #receive = s.recv(4096).decode()
                #print("Received from server:", receive)
                f=2
            f=2
                
            
    except Exception as e:
        print(f"Error in communication with server: {e}")

    cls[0]=100
    cls[1]=100
    cls[2]=100
    cls[3]=100
    #f=0


# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()

# Release the video capture object and close the display window
#cap.release()
cv2.destroyAllWindows()