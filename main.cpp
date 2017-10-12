#include "classifier.h"
#include "extracter.h"
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
	_chdir("MatchNet");

	string modelFolder = "models";

	string trained_filename = modelFolder + "/feature_net.pbtxt";
	string model_filename = modelFolder + "/liberty_r_0.01_m_0.feature_net.pb";

	string output_filename = "bmp.csv";

	Extracter extracter(trained_filename, model_filename, output_filename);

	string imageFolder = "resize_bmp";

	struct dirent *drnt;
	DIR *dr;
	dr = opendir(imageFolder.c_str());
	vector<string> testImages;
	while (drnt = readdir(dr))
	{
		if (drnt->d_type == DT_REG)
		{
			testImages.push_back(drnt->d_name);

		}
	}

	_chdir(imageFolder.c_str());

	for (m = 0; m < testImages.size(); m++)
	{
		std::cout << m + 1 << "/" << testImages.size() << std::endl;

		extracter.featureExtract(testImages[m]);
	}


	return 0;
}