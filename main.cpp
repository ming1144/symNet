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

struct pairPatch{
	int patchID;
	int targetID;
	float Score;
};

int main()
{
	int m;

	_chdir("SymNet");
	_chdir("MatchNet");

	string modelFolder = "models";

	string feature_trained_filename = modelFolder + "/feature_net.pbtxt";
	string feature_model_filename = modelFolder + "/liberty_r_0.01_m_0.feature_net.pb";

	string classifier_trained_filename = modelFolder + "/classifier_net.pbtxt";
	string classifier_model_filename = modelFolder + "/liberty_r_0.01_m_0.classifier_net.pb";

	Extracter extracter(feature_trained_filename, feature_model_filename);
	Classifier classifier(classifier_trained_filename, classifier_model_filename);

	string output_filename = "nonSymmetry.csv";

	std::ofstream output(output_filename);

	string imageFolder = "nonSymmetry";

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
	int patchNum = patch_w * patch_h;
	int pairNum = (patch_w + 1) * patch_w / 2;

	for (m = 0; m < testImages.size(); m++)
	{
		std::cout << m + 1 << "/" << testImages.size() << std::endl;

		Mat img = imread(testImages[m]);
		Mat img_mirror;

		flip(img, img_mirror, 1);

		int ROI_height = img.rows / patch_h;
		int ROI_width = img.cols / patch_w;

		float** features, **features_mirror;

		features = new float*[patchNum];
		features_mirror = new float*[patchNum];

		for (int i = 0; i < patchNum; i++)
		{
			features[i] = new float[4096];
			features_mirror[i] = new float[4096];
		}

		Point *patches = new Point[patchNum];

		for (int i = 0; i < patch_h; i++)
		{
			for (int j = 0; j < patch_w; j++)
			{
				Rect ROI(0 + j*ROI_width, 0 + i*ROI_height, ROI_width, ROI_height);
				Mat patch = img(ROI);
				Mat patch_mirror = img_mirror(ROI);

				patches[j + i*patch_h] = Point(ROI_width/2 + j*ROI_width, ROI_height/2 + i*ROI_height);

				extracter.featureExtract(patch, features[i*patch_h + j]);
				extracter.featureExtract(patch_mirror, features_mirror[i*patch_h + j]);
			}
		}

		float** score;

		score = new float*[pairNum];

		for (int i = 0; i < pairNum; i++)
		{
			score[i] = new float[patch_h];
		}

		output << testImages[m];

		float* scoreSum;
		scoreSum = new float[pairNum];
		memset(scoreSum, 0, pairNum * sizeof(float));

		for (int i = 0, n = 0; i < patch_w; i++)
		{
			for (int j = 0 ; j < patch_w - i; j++, n++)
			{
				for (int x = 0; x < patch_h; x++)
				{
					score[n][x] = classifier.featureCompare(features[j + x*patch_h], features_mirror[j + x*patch_h]);
					scoreSum[n] += score[n][x];
				}
			}
		}

		/*for (int i = 0; i < pairNum; i++)
		{
			output << "," << scoreSum[i];
		}

		output << std::endl;*/

		float threshold = 0.5;
		struct pairPatch* pair;
		pair = new struct pairPatch[patchNum];

		for (int i = 0; i < pairNum; i++)
		{
			//pair[i].patchID = i % patch_w;
			pair[i].targetID = -1;
			pair[i].Score = threshold;
		}

		for (int i = 0; i < patch_h; i++)
		{
			int breakPoint = 0;
			int n = 0;
			for (int j = 0; j < patch_w; j++)
			{
				for (;n < breakPoint;n++)
				{
					if (score[n][i] < pair[i*patch_h + j].Score)
					{
						pair[i].targetID;
						pair[i].Score = score[n][i];
					}
				}
				breakPoint += patch_w - j;
			}
		}


	}


	return 0;
}