import numpy as np
import cv2

# 声明画布全白
canvas = np.ones((2338, 1700, 3),np.uint8)*255
y,x = canvas.shape[:2] #注意是y,x.不是x,y
#定义函数处理图片
def crop(pic):
    # 读入黑背景下的彩色手写数字
    img = cv2.imread(pic)
    # 转换为gray灰度图
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,30,30)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2) #膨胀
    imgThres = cv2.erode(imgDial,kernel,iterations=1) #腐蚀

    contours, hier = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:] # 寻找轮廓

    for cidx,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area>500: #面积大于500才处理
            minAreaRect = cv2.minAreaRect(cnt)
            ((cx, cy), (w, h), theta) = minAreaRect #中心点坐标，长宽，旋转角度[-90,0)，当矩形水平或竖直时均返回-90
            # 转换为整数点集坐标
            rectCnt = np.int64(cv2.boxPoints(minAreaRect))
            # 绘制多边形
#             cv2.polylines(img=canvas, pts=[rectCnt], isClosed=True, color=(0,0,255), thickness=3)

#             cv2.imwrite("front_canvas.png", canvas)
            width,height=int(w),int(h) #转换宽,高为整形
            src_pts = rectCnt.astype("float32")
            # coordinate of the points in box points after the rectangle has been
            # straightened
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            croped = cv2.warpPerspective(img, M, (width, height))
    return croped

def warp(imread,bias=0): # 定义覆盖函数
    roiHeiht,roiWidth = imread.shape[:2] #roi的高,宽
    warpY,warpX=int((x - roiWidth)//2),int(0.1*y)+bias #crop图片插入的元点
    canvas[warpX:warpX + roiHeiht,warpY:warpY + roiWidth] = imread
    
cropFront = crop('front.jpg') #获取front的crop
cropBack = crop('back.jpg') #获取back的crop
warp(cropFront) #替换front_roi
warp(cropBack,bias=int(0.8*y-cropBack.shape[0])) #替换back_roi

cv2.imwrite('sfzFinal.jpg',canvas)
