// Step 3
#include <fstream>
#include <utility>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"

using namespace std;
using namespace cv;

#define INPUT_W 474
#define INPUT_H 474
#define num_classes 4

// Define a function to convert Opencv's Mat data to tensorflow tensor. In python, just perform np.reshape on the matrix read in cv2.imread(), and the data type becomes a tensor, that is, tensor is the same as a matrix. Then you can even enter the entrance of the network
// In the C++ version, the input of the network also needs to be a tensor data type, so the input image needs to be converted into a tensor. If you use Opencv to read the image, the format is a Mat, you need to consider how to convert a Mat to tensor
void CVMat_to_Tensor(cv::Mat& img, tensorflow::Tensor& output_tensor, int input_rows, int input_cols)
{
	Mat resize_img;
	resize(img, resize_img, cv::Size(input_cols, input_rows));

	Mat dst = resize_img.reshape(1, 1);
	// 第二个坑 rgb图像的归一化
	for (int i = 0; i < dst.cols; i++) {
		dst.at<float>(0, i) = dst.at<float>(0, i) / 255.0;
	}
	resize_img = dst.reshape(3, INPUT_H);

	float * p = (&output_tensor)->flat<float>().data();
	cv::Mat tempMat(input_rows, input_cols, CV_32FC3, p);
	resize_img.convertTo(tempMat, CV_32FC3);

}

void tensor2Mat(tensorflow::Tensor &t, cv::Mat &image) {
	float *p = t.flat<float>().data();
	image = Mat(INPUT_H, INPUT_W, CV_32FC3, p); //根据分类个数来，现在是3分类，如果是4分类就写成CV_32FC4

}


int main(int argc, char ** argv)
{
	/* --------------------Configuration key information------------------------- -----------*/
	std::string model_path = "./psp_1w_resnet50_wudongjie.pb"; // pb model address
	std::string image_path = "./model/image.jpg"; // test picture
	int input_height = INPUT_H; // Enter the image height of the network
	int input_width = INPUT_W; // input network image width
	std::string input_tensor_name = "input_1:0"; // The name of the input node of the network
	std::string output_tensor_name = "main/truediv:0"; // The name of the output node of the network

	/* --------------------Create session------------------------------------*/
	tensorflow::Session * session;
	tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session); // Create a new session Session

	/* --------------------Read model from pb file------------------------------------*/
	tensorflow::GraphDef graphdef; //Define a graph for the current model
	tensorflow::Status status_load = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path, &graphdef); // read graph model from pb file
	if (!status_load.ok()) // Determine whether the read model is correct, if it is wrong, print out the wrong information
	{
		std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
		std::cout << status_load.ToString() << "\n";
		return -1;
	}

	tensorflow::Status status_create = session->Create(graphdef); // Import the model into the session Session
	if (!status_create.ok()) // Determine whether the model is imported into the session successfully, if it is wrong, print out the error message
	{
		std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
		return -1;
	}
	std::cout << "<------Sucessfully created session and load graph------>" << std::endl;

	/* --------------------Load test picture------------------------ ------------*/
	cv::Mat img = cv::imread(image_path, -1); // read image, read grayscale image
	img.convertTo(img, CV_32FC3);//第一个小坑，整个程序都读取Float

	if (img.empty())
	{
		std::cout << "can't open the image!!!!!" << std::endl;
		return -1;
	}
	// Create a tensor as the input network interface
	tensorflow::Tensor resized_tensor(tensorflow::DT_FLOAT, 		    tensorflow::TensorShape({1, input_height,input_width, 3}));

	// Save the Mat format picture read by opencv into tensor
	CVMat_to_Tensor(img, resized_tensor, input_height, input_width);
	std::cout << resized_tensor.DebugString() << std::endl;

	/* --------------------Test with network------------------------ ------------*/
	std::cout << std::endl << "<------------------Runing the model with test_image------------------->" << std::endl;
	// Run forward, the output result must be a vector of tensor

	std::vector<tensorflow::Tensor> outputs;
	std::string output_node = output_tensor_name; // output node name
	tensorflow::Status status_run = session->Run({ { input_tensor_name, resized_tensor } }, { output_node }, {}, &outputs);
	if (!status_run.ok())
	{
		std::cout << "ERROR: Run failed..." << std::endl;
		std::cout << status_run.ToString() << std::endl;
		return -1;
	}
	// Extract the output value
	std::cout << "Output tensor size: " << outputs.size() << std::endl;
	for (std::size_t i = 0; i < outputs.size(); i++)
	{
		std::cout << outputs[i].DebugString() << std::endl;
	}
	tensorflow::Tensor t = outputs[0];

	cv::Mat outimage = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
	cv::Mat class_Mat;
	tensor2Mat(t, class_Mat);

	int output_height = t.shape().dim_size(1);
	int output_width = t.shape().dim_size(2);
	
	//根据分类数目设置颜色个数
	int colors[num_classes][3] = { { 73, 73, 73 },{ 0, 255, 255 },{ 255, 255, 0 } };
	
	//根据one-hot编码实现每个像素的分类，选取三通道中概率最大值最为分类结果，并赋予颜色
	for (int i = 0; i < output_height; i++)
	{
		for (int j = 0; j < output_width; j++)
		{
			int index = 0;
			for (int k = 1; k < num_classes; k++) {				
				if ((float)class_Mat.at<Vec3f>(i, j)[k] >(float)class_Mat.at<Vec3f>(i, j)[k - 1]) {//3分类所以是Vec3f
					index = k;
				}
			}
			
			if (index ==0) {
				outimage.at<Vec3f>(i, j)[0] = colors[0][0];
				outimage.at<Vec3f>(i, j)[1] = colors[0][1];
				outimage.at<Vec3f>(i, j)[2] = colors[0][2];
			}
			else if (index == 1) {
				outimage.at<Vec3f>(i, j)[0] = colors[1][0];
				outimage.at<Vec3f>(i, j)[1] = colors[1][1];
				outimage.at<Vec3f>(i, j)[2] = colors[1][2];
			}
			else {
				outimage.at<Vec3f>(i, j)[0] = colors[2][0];
				outimage.at<Vec3f>(i, j)[1] = colors[2][1];
				outimage.at<Vec3f>(i, j)[2] = colors[2][2];
			}
		}
	}
	// 记得最后要resize到原图像大小
	resize(outimage, outimage, cv::Size(img.cols, img.rows));
	imshow("img", outimage);
	waitKey();

	return 0;
}

