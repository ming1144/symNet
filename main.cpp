#include "classifier.h"
#include "register.h"
#include "imageTransform.h"

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <direct.h>

#include <dirent.h>

#include <omp.h>


#include <caffe\caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <QtGui\QImage>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

int maskHeight = 33;
bool padding = false;
bool withOriginImage = false;

int main()
{
	int m;
	
	_chdir("tumor");

	string modelFolder = "20170804-networkInNetwork";

	string trained_filename = modelFolder + "/deploy.prototxt";
	string mean_filename = modelFolder + "/mean.binaryproto";
	string model_filename = modelFolder + "/model.caffemodel";
	string label_filename = modelFolder + "/labels.txt";


	Classifier classifier(trained_filename, model_filename, mean_filename, label_filename);

	//_chdir("train");
	//_chdir("test");	

	struct dirent *drnt;
	DIR *dr;
	dr = opendir("origin");
	vector<string> testImages;
	while (drnt = readdir(dr))
	{
		if (drnt->d_type == DT_REG)
		{
			testImages.push_back(drnt->d_name);

		}
	}
	_chdir("origin");
	_mkdir(modelFolder.c_str());

//#pragma omp parallel for
	for (m = 0; m < testImages.size(); m++)
	{
		std::cout << testImages[m] << " " << m+1 << "/" << testImages.size() << std::endl;


		int i, j;
		Mat testImg = cv::imread(testImages[m]);
		Mat testImgGray(testImg);
		cvtColor(testImg, testImgGray, CV_BGR2GRAY);		
		
		Mat LBP(testImg.rows, testImg.cols, CV_8UC1, Scalar(0)); 
		Mat nakagami(testImg.rows, testImg.cols, CV_8UC1, Scalar(0));

		//imageTransform::localBinaryPattern(&testImgGray, &LBP);
		//imageTransform::nakagami(&testImg,&nakagami);
		vector<Mat> channel;
		channel.push_back(testImgGray);
		//channel.push_back(LBP);
		//channel.push_back(nakagami);
		Mat combineImage;
		merge(channel, combineImage);


		Mat result;
		if (!withOriginImage)
		{
			result = Mat(testImg.rows, testImg.cols, testImg.type(), Scalar(0));
		}
		else
		{
			result = Mat(testImg);
		}
		//Mat tempImg(testImgGray.rows + maskHeight - 1, testImgGray.cols + maskHeight - 1, testImgGray.type(), Scalar(0));
		Mat tempImg(combineImage.rows + maskHeight - 1, combineImage.cols + maskHeight - 1, combineImage.type(), Scalar(0));

		combineImage.copyTo(tempImg.rowRange(maskHeight / 2, maskHeight / 2 + combineImage.rows).colRange(maskHeight / 2, maskHeight / 2 + combineImage.cols));
		for (i = maskHeight / 2; i < tempImg.rows - maskHeight / 2; i++)
		{
//#pragma omp parallel for
			for (j = maskHeight / 2; j < tempImg.cols - maskHeight / 2; j++)
			{
				Point truePoint;
				truePoint.x = j - maskHeight / 2;
				truePoint.y = i - maskHeight / 2;
				if ( testImgGray.at<uchar>(truePoint) <= 10 )
				{
					result.at<Vec3b>(truePoint) = Vec3b::all(0);
					continue;
				}
				Rect Region(Point(j - maskHeight / 2, i - maskHeight / 2), Point(j + maskHeight / 2 + 1, i + maskHeight / 2 + 1));
				Mat ROI(tempImg, Region);

				std::vector<Prediction> predictions = classifier.Classify(ROI);

				Prediction temp = predictions[0];
				if (temp.first != "0")
				{
					result.at<Vec3b>(truePoint) = Vec3b::all(255);
				}
			}
		}

		string resultFilename = modelFolder + "/";
		resultFilename += testImages[m];
		cv::imwrite(resultFilename, result);
	}
	

	return 0;
}