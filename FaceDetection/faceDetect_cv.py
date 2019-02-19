
import os
import cv2
import time

def detect_impl(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color = (0, 255, 0)

    # OpenCV人脸识别分类器
    classifier_file = os.path.join(os.getcwd(), '../data/haarcascade_frontalface_default.xml')
    # print(os.path.isfile(classifier_file))

    classifier = cv2.CascadeClassifier(classifier_file)

    # 调用识别人脸
    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # # 左眼
            # cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            # # 右眼
            # cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            # # 嘴巴
            # cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color)

def detect_img():
        '''
        图片转换成灰色（去除色彩干扰，让图片识别更准确）
        图片上画矩形
        使用训练分类器查找人脸
        '''
        file = os.path.join(os.getcwd(), '../test-images/timg3.jpg')
        b = os.path.isfile(file)
        # print(b)

        img = cv2.imread(file)

        cv2.namedWindow('image')

        detect_impl(img)

        cv2.imshow("image", img)  # 显示图像
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_video():
    # 获取摄像头0表示第一个摄像头
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    time0 = time.time()
    frmCnt = 0
    while (1):  # 逐帧显示
        ret, img = cap.read()
        detect_impl(img)
        cv2.imshow("image", img)
        frmCnt += 1
        time1 = time.time()
        if (time1 - time0 >= 1.0):
            print(frmCnt)
            time0 = time1
            frmCnt = 0

        # 获取用户输入的最后一个字符的ASCII码，如果输入的是“q”，则跳出循环。
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 释放窗口资源

if __name__ == '__main__':
    # detect_img()
    detect_video()