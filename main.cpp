#define CPU_ONLY

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

	string output_filename = "symmetry.csv";

	std::ofstream output(output_filename);

	Extracter extracter(trained_filename, model_filename, output_filename);

	string imageFolder = "symmetry";

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

	int patch_w = 8;
	int patch_h = 4;


	for (m = 0; m < testImages.size(); m++)
	{
		std::cout << m + 1 << "/" << testImages.size() << std::endl;

		Mat img = imread(testImages[m]);
		Mat img_mirror;

		flip(img, img_mirror, 1);

		

		int ROI_height = img.rows / patch_h;
		int ROI_width = img.cols / patch_w;

		float** features, **features_mirror;

		features = new float*[patch_h*patch_w];
		features_mirror = new float*[patch_h*patch_w];

		for (int i = 0; i < patch_h*patch_w; i++)
		{
			features[i] = new float[4096];
			features_mirror[i] = new float[4096];
		}


		for (int i = 0; i < patch_h; i++)
		{
			for (int j = 0; j < patch_w; j++)
			{
				Rect ROI(0 + j*ROI_width, 0 + i*ROI_height, ROI_width, ROI_height);
				Mat patch = img(ROI);
				Mat patch_mirror = img_mirror(ROI);

				extracter.featureExtract(patch, features[i*patch_h + j]);
				extracter.featureExtract(patch_mirror, features_mirror[i*patch_h + j]);
			}
		}


		
	}


	return 0;
}