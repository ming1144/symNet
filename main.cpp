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

int main()
{
	int m;
	
	_chdir("SymNet");

	string modelFolder = "2channel";

	string trained_filename = modelFolder + "/deploy.prototxt";
	string model_filename = modelFolder + "/model.caffemodel";
	string label_filename = modelFolder + "/labels.txt";

	Classifier classifier(trained_filename, model_filename, label_filename);

	struct dirent *drnt;
	DIR *dr;
	dr = opendir("car");
	vector<string> testImages;
	while (drnt = readdir(dr))
	{
		if (drnt->d_type == DT_REG)
		{
			testImages.push_back(drnt->d_name);

		}
	}

	std::ofstream output;
	output.open(modelFolder + ".csv");

//#pragma omp parallel for
	for (m = 0; m < testImages.size(); m++)
	{
		std::cout << m + 1 << "/" << testImages.size() << std::endl;


		Mat testImg = cv::imread("car/"+testImages[m]);

		vector<Prediction> prediction = classifier.Classify(testImg);		

 		output << testImages[m] << "," << prediction[0].second << "," << prediction[1].second << std::endl;
	}
	

	return 0;
}