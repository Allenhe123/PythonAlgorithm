
'''
识别精准度：Dlib >= OpenCV
Dlib更多的人脸识别模型，可以检测脸部68甚至更多的特征点
Dlib的人脸识别要比OpenCV精准很多，一个是模型方面的差距，在一方面和OpenCV的定位有关系，OpenCV是一个综合性的视觉处理库
1.Dlib模型识别的准确率和效果要好于OpenCV；
2.Dlib识别的性能要比OpenCV差，使用视频测试的时候Dlib有明显的卡顿，但是OpenCV就好很多，基本看不出来；

'''
import os
import time
import cv2
import dlib

def detect_impl(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸分类器
    detector = dlib.get_frontal_face_detector()
    # 获取人脸检测器
    detector_file = os.path.join(os.getcwd(), '../data/shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(detector_file)

    dets = detector(gray, 1)
    for face in dets:
        shape = predictor(img, face)  # 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)

def detect_img():
    file = os.path.join(os.getcwd(), '../test-images/timg3.jpg')
    img = cv2.imread(file)

    detect_impl(img)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video():
    # 获取摄像头0表示第一个摄像头
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    time0 = time.time()
    frmCnt = 0
    detector = dlib.get_frontal_face_detector()  # 使用默认的人类识别器模型
    while (1):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ######
        dets = detector(gray, 1)
        for face in dets:
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            # cv2.imshow("image", img)
        ######
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