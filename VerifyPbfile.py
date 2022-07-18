# Step 2
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
//INPUT_TENSOR_NAME和OUTPUT_TENSOR_NAME 是基于1中的结果
INPUT_TENSOR_NAME = 'input_1:0'
OUTPUT_TENSOR_NAME = 'main/truediv:0'
INPUT_SIZE = 473
colors = [(73, 73, 73), (0, 255, 255), (255, 255, 0)]
num_classes = 3

with tf.gfile.FastGFile('./model.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def, name="")
sess = tf.Session(graph=g_in)


def run(image):
    height, width = image.shape[0:2]
    # 归一化很重要，卡了我1天多
    image = (image - image.min())/(image.max() - image.min())
    resized_image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

    input_x = sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME)
    out_softmax = sess.graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)
    
    batch_seg_map = sess.run(
        out_softmax,
        feed_dict={input_x: [np.asarray(resized_image)]})
   # batch_seg_map是[1,473,473,3]，batch_seg_map[0]尺寸是[473,473,3]
   # seg_map 便是预测结果
    seg_map = batch_seg_map[0]
    seg_map = seg_map.argmax(axis=-1).reshape([INPUT_SIZE, INPUT_SIZE])
    
    seg_img = np.zeros((np.shape(seg_map)[0], np.shape(seg_map)[1], 3))
	# 根据seg_map 中每个像素的值来赋予mask颜色
    for c in range(num_classes):
        seg_img[:, :, 0] += ((seg_map[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_map[:, :] ==c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_map[:, :] == c) * (colors[c][2])).astype('uint8')

    image = cv2.resize(seg_img, (int(width), int(height)))
    return image


input_image = cv2.imread('./img/image.jpg')
seg_map = run(input_image)
cv2.imwrite("./out.jpg", seg_map)
