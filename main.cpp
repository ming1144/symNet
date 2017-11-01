#include "register.h"
#include "imageTransform.h"
#include "symNet.h"

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

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;


int main()
{
	int i, j, m, p, x, y;

	_chdir("SymNet");
	_chdir("MatchNet");

	string modelFolder = "models";

	string feature_trained_filename = modelFolder + "/feature_net.pbtxt";
	string feature_model_filename = modelFolder + "/liberty_r_0.01_m_0.feature_net.pb";

	string classifier_trained_filename = modelFolder + "/classifier_net.pbtxt";
	string classifier_model_filename = modelFolder + "/liberty_r_0.01_m_0.classifier_net.pb";

	symNet Net(feature_trained_filename, feature_model_filename, classifier_trained_filename, classifier_model_filename);

	//_chdir("indoor");
	string testFolder = "ToyotaAltis_2010";
	_chdir(testFolder.c_str());

	string imageFolder = "image";

	//string coordinate_filename = "symmeletCoordinate.csv";

	string rootFolder = ".";

	Net.setThreshold(0.9);

	//Net.singleImage(rootFolder, testFolder);
	Net.slidingWindowDetect(rootFolder, imageFolder);
	//Net.symSURFDetect(rootFolder, coordinate_filename);

	return 0;
}