#define CPU_ONLY

#include "register.h"
#include "imageTransform.h"
#include "symNet.h"
#include "selectiveSearch.h"

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <direct.h>
#include <dirent.h>

#include <caffe\caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;


int main()
{
	_chdir("SymNet");
	_chdir("SymNet");

	string modelFolder = "models";
	string modelDate = "20180101";

	string classifier_trained_filename = modelFolder + '/' + modelDate + "/deploy.prototxt";
	string classifier_model_filename = modelFolder + '/' + modelDate + "/model.caffemodel";
	string classifier_label_filename = modelFolder + "/labels.txt";

	vector<int> mean;
	mean.push_back(101);
	mean.push_back(107);
	mean.push_back(112);
	float variance = 4741.28;

	//Classifier classifier(classifier_trained_filename, classifier_model_filename, classifier_label_filename, mean, variance);
	Classifier classifier(classifier_trained_filename, classifier_model_filename, classifier_label_filename);

	//string posFolder = "pos";
	//vector<string> posList;
	//struct dirent* dirt;
	//DIR* dir = opendir(posFolder.c_str());
	//while (dirt = readdir(dir))
	//{
	//	if (dirt->d_type == DT_REG)
	//	{
	//		posList.push_back(dirt->d_name);
	//	}
	//}

	//string negFolder = "neg";
	//vector<string> negList;
	//dir = opendir(negFolder.c_str());
	//while (dirt = readdir(dir))
	//{
	//	if (dirt->d_type == DT_REG)
	//	{
	//		negList.push_back(dirt->d_name);
	//	}
	//}

	//_mkdir(modelDate.c_str());
	//_chdir(modelDate.c_str());
	//_mkdir("pos");
	//_chdir("pos");
	//_mkdir("yes");
	//_mkdir("no");
	//ofstream pos(modelDate + ".csv", ios::trunc);
	//_chdir("..");

	//_mkdir("neg");
	//_chdir("neg");
	//_mkdir("yes");
	//_mkdir("no");
	//ofstream neg(modelDate + ".csv", ios::trunc);
	//_chdir("..");

	//Mat img = imread(posFolder + '/' + posList[0], CV_LOAD_IMAGE_COLOR);
	//_chdir("..");
	//img = imread(posFolder + '/' + posList[0], CV_LOAD_IMAGE_COLOR);
	//
	//pos << "file,Yes,No\n";
	//neg << "file,Yes,No\n";

	//int count = 0;
	//for (int m = 0; m < posList.size(); m++)
	//{
	//	Mat img = imread(posFolder + '/' + posList[m], CV_LOAD_IMAGE_COLOR);
	//	vector<Prediction> predict = classifier.Classify(img);

	//	pos << posList[m] << ',';
	//	if (!predict[0].first.compare("1"))
	//	{
	//		pos << predict[0].second << ',' << predict[1].second << endl;
	//		count++;
	//		imwrite(modelDate + "/pos/yes/" + posList[m], img);
	//	}
	//	else
	//	{
	//		pos << predict[1].second << ',' << predict[0].second << endl;
	//		imwrite(modelDate + "/pos/no/" + posList[m], img);
	//	}
	//	cout << m + 1 << '/' << posList.size() << endl;
	//	//cout << list[m] << "  " << predict[0].first << ":" << predict[0].second << " " << predict[1].first << ":" << predict[1].second << endl;
	//}

	//pos << "Accuracy," << (float)count / posList.size();

	//count = 0;
	//for (int m = 0; m < negList.size(); m++)
	//{
	//	Mat img = imread(negFolder + '/' + negList[m], CV_LOAD_IMAGE_COLOR);
	//	vector<Prediction> predict = classifier.Classify(img);

	//	neg << negList[m] << ',';
	//	if (!predict[0].first.compare("1"))
	//	{
	//		neg << predict[0].second << ',' << predict[1].second << endl;
	//		imwrite(modelDate + "/neg/yes/" + negList[m], img);
	//	}
	//	else
	//	{
	//		neg << predict[1].second << ',' << predict[0].second << endl;
	//		imwrite(modelDate + "/neg/no/" + negList[m], img);
	//		count++;
	//	}
	//	cout << m + 1 << '/' << negList.size() << endl;
	//	//cout << list[m] << "  " << predict[0].first << ":" << predict[0].second << " " << predict[1].first << ":" << predict[1].second << endl;
	//}

	//neg << "Accuracy," << (float)count / negList.size();

	string Folder = "image";
	vector<string> list;
	struct dirent* dirt;
	DIR* dir = opendir(Folder.c_str());
	while (dirt = readdir(dir))
	{
		if (dirt->d_type == DT_REG)
		{
			list.push_back(dirt->d_name);
		}
	}
	for (int m = 0; m < list.size(); m++)
	{
		Mat img = imread(Folder + "/" + list[m]);

		vector<Rect> ROI = ss::selectiveSearch(img);

		int j;
	}

	return 0;
}