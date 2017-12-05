#include "classifier.h"
#include "extracter.h"
#include "NMS.h"
#include <direct.h>
#include <dirent.h>

#include <iostream>

struct symSURFImage
{
	std::string filename;
	int symSURFPairNum;
	cv::Point* center, *left, *right;
};

class symNet
{
public:
	symNet(string& feature_trained_filename, string& feature_model_filename, string& classifier_trained_filename, string& classifier_model_filename)
	{
		extracter = Extracter::Extracter(feature_trained_filename, feature_model_filename);
		classifier = Classifier::Classifier(classifier_trained_filename, classifier_model_filename);
	};
	~symNet(){};
	void symSURFDetect(string& root, string& file);
	void slidingWindowDetect(string& root, string& folder);
	void singleImage(string&root, string& file);
	void setROI(int width, int height)
	{
		ROI_width = width;
		ROI_height = height;
	};
	void setStep(int step_new)
	{
		step = step_new;
	};
	void setPatch(int width, int height)
	{
		patch_w = width;
		patch_h = height;
		patchNum = width*height;
	};
	void setThreshold(float threshold_new)
	{
		threshold = threshold_new;
	};
	void CreateROI(bool option)
	{
		createROI = option;
	};
	void UseNMS(bool option)
	{
		useNMS = option;
	};
private:
	void readSymSURFPair(string& root, string& filename);
	void readDirectory(string& root, string& folder);
	Extracter extracter;
	Classifier classifier;
	vector<struct symSURFImage> list;


	int step = 20;
	int ROI_height = 140;
	int ROI_width = 64;
	int patch_w = 1;
	int patch_h = 3;
	int patchNum = patch_w * patch_h;
	float threshold = 0.7;

	bool createROI = false;
	bool useNMS = true;
};