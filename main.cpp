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
	string modelDate = "20171211";
	bool crossEntropy = true;

	string feature_trained_filename = modelFolder + "/feature_net.pbtxt";
	string feature_model_filename = modelFolder + "/liberty_r_0.01_m_0.feature_net.pb";

	string classifier_trained_filename = modelFolder + "/classifier_net.pbtxt";
	string classifier_model_filename = modelFolder + "/liberty_r_0.01_m_0.classifier_net.pb";

	string symClassifier_trained_filename;
	string symClassifier_model_filename;
	string symClassifier_label_filename;
	if (!crossEntropy)
	{
		symClassifier_trained_filename = modelFolder + '/' + modelDate + "/symNet.prototxt";
		symClassifier_model_filename = modelFolder + '/' + modelDate + "/symNet.caffemodel";
		symClassifier_label_filename = modelFolder + "/labels.txt";
	}
	else
	{
		symClassifier_trained_filename = modelFolder + '/' + modelDate + "-CrossEntropy/symNet.prototxt";
		symClassifier_model_filename = modelFolder + '/' + modelDate + "-CrossEntropy/symNet.caffemodel";
		symClassifier_label_filename = modelFolder + "/labels_CrossEntropy.txt";
	}

	symNet Net(feature_trained_filename, feature_model_filename,
		classifier_trained_filename, classifier_model_filename,
		symClassifier_trained_filename, symClassifier_model_filename, symClassifier_label_filename,
		modelDate);

	//_chdir("indoor");
	string testFolder = "ToyotaAltis_2010";
	_chdir(testFolder.c_str());

	string imageFolder = "image";

	string coordinate_filename = "symmeletCoordinate.csv";

	string rootFolder = ".";

	Net.setThreshold(0.8);
	Net.setPatch(1, 1);
	Net.setROI(64, 64);
	Net.setStep(5);
	Net.CreateROI(true);
	Net.UseCrossEntropy(true);

	string testImage = "1.bmp";
	Net.singleImage(imageFolder, testImage);

	//Net.slidingWindowDetect(rootFolder, imageFolder);
	//Net.symSURFDetect(rootFolder, coordinate_filename);

	return 0;
}