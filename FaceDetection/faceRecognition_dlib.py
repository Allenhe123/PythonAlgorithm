import sys, os, dlib, glob
import numpy as np
from skimage import io

'''
http://dlib.net/
http://dlib.net/python/
'''

print('the battle begin...')

# 1.人脸关键点检测器   shape_predictor_68_face_landmarks.dat是已经训练好的人脸关键点检测器。
predictor_path =  '../data/shape_predictor_68_face_landmarks.dat'

# 2.人脸识别模型  dlib_face_recognition_resnet_model_v1.dat是训练好的ResNet人脸识别模型
face_rec_model_path = '../data/dlib_face_recognition_resnet_model_v1.dat'

# 3.候选人脸文件夹
faces_folder_path = '../training-images'

# 4.需识别的人脸
img_path = '../test-images'

# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()  # Returns the default face detector

# 2.加载人脸关键点检测器
'''
This object is a tool that takes in an image region containing some object and outputs a set of point locations
that define the pose of the object. The classic example of this is human face pose prediction, where you take an image
of a human face as input and are expected to identify the locations of important facial landmarks such as the corners 
of the mouth and eyes, tip of the nose, and so forth.
'''
sp = dlib.shape_predictor(predictor_path)    # class dlib.shape_predictor

# 3. 加载人脸识别模型
'''
class dlib.face_recognition_model_v1 - This object maps human faces into 128D vectors
where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart.
'''
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

# 候选人脸描述子list
descriptors = []


# 对文件夹下的每一个人脸进行:
# 1.人脸检测
# 2.关键点检测
# 3.描述子提取

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    win.clear_overlay()
    win.set_image(img)

    # 1.人脸检测
    dets = detector(img, 1)   # return class dlib.rectangles -- An array of rectangle objects.
    print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):  #  enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        # 2.关键点检测
        shape = sp(img, d)        # d is dlib.rectangle object type
        # 画出人脸区域和和关键点
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)

        # 3.描述子提取，128D向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)   # return dlib.vector -- This object represents the mathematical idea of a column vector.

        # 转换为numpy array
        v = np.array(face_descriptor)
        descriptors.append(v)


# 对需识别人脸进行同样处理
# 提取描述子
for f in glob.glob(os.path.join(img_path, "*.jpg")):
    print("\n Testing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img, 1)

    dist = []
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = np.array(face_descriptor)

        # 计算欧式距离
        for i in descriptors:
            # linalg=linear（线性）+algebra（代数） linalg.norm()用于求范数。默认是2范数，也就是欧式距离
            dist_ = np.linalg.norm(i - d_test)
            dist.append(dist_)

    # 候选人名单
    candidate = ['高圆圆','马伊利','张柏芝','李小璐','范冰冰','钟欣桐', '刘亦菲']

    # 候选人和距离组成一个dict
    c_d = dict(zip(candidate, dist))
    cd_sorted = sorted(c_d.items(), key=lambda d:d[1])  # 根据dist中的距离由小到大排序c_d.items()这个list
    print("The person is: ", cd_sorted[0][0])
    print(" distance is: ", cd_sorted[0][1])

dlib.hit_enter_to_continue()
