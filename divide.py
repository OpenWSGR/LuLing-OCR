# running: 

import cv2
import numpy as np
import os
import sys

# # 检查文件是否存在
# image_path = r'D:\github\LuLing-OCR\yuan.png'
# if not os.path.exists(image_path):
#     print(f"Error: File not found at {image_path}")
# else:
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Unable to load image.")
#     else:
#         cv2.imshow('Image', image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
# 加载图像

def process_images(ori_path, dst_path):
    # 检查输入目录是否存在
    if not os.path.exists(ori_path):
        print(f"Error: Input directory '{ori_path}' does not exist.")
        return

    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        print(f"Created output directory '{dst_path}'.")




    # #with open(dst_path, 'w') as op:
    # # 遍历输入目录中的所有 .png 文件
    # for filename in os.listdir(ori_path):
    #     if filename.endswith('.png'):
    #         # 构造完整的文件路径
    #         input_file_path = os.path.join(ori_path, filename)
    #         output_file_path = os.path.join(dst_path, filename)
    #         # 读取图像
    #         img = cv2.imread(input_file_path)
    #         if img is None:
    #             if img is None:
    #                 print(f"Error: Unable to load image '{input_file_path}'.")
    #                 continue
    #         # 转换为灰度图像
    #         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         # 应用二值化
    #         retval, threshold_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV)
    #         cv2.imshow("threshold_img", threshold_img)
    #         cv2.waitKey(0)
    #         # 水平投影
    #         gray_value_x = []
    #         sz1, sz2 = threshold_img.shape
    #         for i in range(sz1):
    #             white_value = np.sum(threshold_img[i, :] == 255)
    #             gray_value_x.append(white_value)
    #         # 创建水平投影图像
    #         hori_projection_img = np.zeros((sz1, sz2), np.uint8)
    #         for i in range(sz1):
    #             hori_projection_img[i, :gray_value_x[i]] = 255
    #         cv2.imshow("hori_projection_img", hori_projection_img)
    #         cv2.waitKey(0)
    #         # 检测行
    #         text_rect_x = []
    #         inline_x = 0
    #         start_x = 0
    #         for i in range(len(gray_value_x)):
    #         #    print (inline_x)
    #         #    print (gray_value_x[i])
    #             if inline_x == 0 and gray_value_x[i] < 95:
    #                 inline_x = 1
    #                 start_x = i
    #                 # print(inline_x)
    #                 # print(i)
    #             elif inline_x == 1 and gray_value_x[i] >= 95 and (i - start_x) > 5:
    #                 inline_x = 0
    #                 # print (i)
    #                 # print(start_x)
    #                 if i - start_x > 10:
    #                     text_rect_x.append([start_x, i])
    #         print("分行区域，每行数据起始位置Y：", text_rect_x)
    #         #op.write(text_rect_x)

    #         # 垂直投影
    #         gray_value_y = []
    #         for i in range(sz2):
    #             white_value = np.sum(threshold_img[:, i] == 255)
    #             gray_value_y.append(white_value)
    #         # 创建垂直投影图像
    #         veri_projection_img = np.zeros((sz1, sz2), np.uint8)
    #         for i in range(sz2):
    #             veri_projection_img[:gray_value_y[i], i] = 255
    #         cv2.imshow("veri_projection_img", veri_projection_img)
    #         cv2.waitKey(0)
    #         # 检测列
    #         text_rect_y = []
    #         inline_y = 0
    #         start_y = 0
    #         for i in range(len(gray_value_y)):
    #             if inline_y == 0 and gray_value_y[i] < 95:
    #                 inline_y = 1
    #                 start_y = i
    #             elif inline_y == 1 and gray_value_y[i] >= 95 and (i - start_y) > 5:
    #                 inline_y = 0
    #                 if i - start_y > 10:
    #                     text_rect_y.append([start_y, i])
    #         print("分列区域，每列数据起始位置X：", text_rect_y)
    #         #op.write(text_rect_y)
            
    #         # 裁剪文字区域
    #         for rect in text_rect_x:
    #             cropImg = threshold_img[rect[0]:rect[1], 0:sz2]
    #             cv2.imshow("cropImg", cropImg)
    #             cv2.waitKey(0)
    #         for rect in text_rect_x:
    #             for rec in text_rect_y:
    #                 cropImg = threshold_img[rect[0]:rect[1], rec[0]:rec[1]]
    #                 cv2.imshow("cropImg", cropImg)
    #                 cv2.waitKey(0)
                        
    #         # img = cv2.imread(r'D:\github\LuLing-OCR\2.png')
    #         # cv2.imshow("Orig Image", img)

    

    with open(dst_path, 'w') as op:
        # 遍历输入目录中的所有 .png 文件
        for filename in os.listdir(ori_path):
            if filename.endswith('.png'):
                # 构造完整的文件路径
                input_file_path = os.path.join(ori_path, filename)
                output_file_path = os.path.join(dst_path, filename)

                # 读取图像
                img = cv2.imread(input_file_path)
                if img is None:
                    if img is None:
                        print(f"Error: Unable to load image '{input_file_path}'.")
                        continue
                # 转换为灰度图像
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 应用二值化
                retval, threshold_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV)
                cv2.imshow("threshold_img", threshold_img)
                cv2.waitKey(0)

                # 水平投影
                gray_value_x = []
                sz1, sz2 = threshold_img.shape
                for i in range(sz1):
                    white_value = np.sum(threshold_img[i, :] == 255)
                    gray_value_x.append(white_value)

                # 创建水平投影图像
                hori_projection_img = np.zeros((sz1, sz2), np.uint8)
                for i in range(sz1):
                    hori_projection_img[i, :gray_value_x[i]] = 255
                cv2.imshow("hori_projection_img", hori_projection_img)
                cv2.waitKey(0)

                # 检测行
                text_rect_x = []
                inline_x = 0
                start_x = 0
                for i in range(len(gray_value_x)):
                #    print (inline_x)
                #    print (gray_value_x[i])
                    if inline_x == 0 and gray_value_x[i] < 95:
                        inline_x = 1
                        start_x = i
                        # print(inline_x)
                        # print(i)
                    elif inline_x == 1 and gray_value_x[i] >= 95 and (i - start_x) > 5:
                        inline_x = 0
                        # print (i)
                        # print(start_x)
                        if i - start_x > 10:
                            text_rect_x.append([start_x, i])
                print("分行区域，每行数据起始位置Y：", text_rect_x)
                #op.write(text_rect_x)

                # 垂直投影
                gray_value_y = []
                for i in range(sz2):
                    white_value = np.sum(threshold_img[:, i] == 255)
                    gray_value_y.append(white_value)

                # 创建垂直投影图像
                veri_projection_img = np.zeros((sz1, sz2), np.uint8)
                for i in range(sz2):
                    veri_projection_img[:gray_value_y[i], i] = 255
                cv2.imshow("veri_projection_img", veri_projection_img)
                cv2.waitKey(0)
                # 检测列
                text_rect_y = []
                inline_y = 0
                start_y = 0
                for i in range(len(gray_value_y)):
                    if inline_y == 0 and gray_value_y[i] < 95:
                        inline_y = 1
                        start_y = i
                    elif inline_y == 1 and gray_value_y[i] >= 95 and (i - start_y) > 5:
                        inline_y = 0
                        if i - start_y > 10:
                            text_rect_y.append([start_y, i])
                print("分列区域，每列数据起始位置X：", text_rect_y)
                #op.write(text_rect_y)
                
                # 裁剪文字区域
                for rect in text_rect_x:
                    cropImg = threshold_img[rect[0]:rect[1], 0:sz2]
                    cv2.imshow("cropImg", cropImg)
                    cv2.waitKey(0)
                for rect in text_rect_x:
                    for rec in text_rect_y:
                        cropImg = threshold_img[rect[0]:rect[1], rec[0]:rec[1]]
                        cv2.imshow("cropImg", cropImg)
                        cv2.waitKey(0)
                            
                # img = cv2.imread(r'D:\github\LuLing-OCR\2.png')
                # cv2.imshow("Orig Image", img)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <ori_path> <dst_path>")
        sys.exit(1)

    ori_path = sys.argv[1]
    dst_path = sys.argv[2]

    process_images(ori_path, dst_path)