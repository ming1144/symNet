#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <caffe\caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::string;
using std::vector;

using boost::shared_ptr;
using caffe::Net;

class Extracter {
public:
	Extracter(const string& model_file,
		const string& trained_file,
		const string& output_file);

	float** featureExtract(const cv::Mat& img, const int patch_h = 4, const int patch_w = 8);

private:
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;

	std::ofstream output;
	cv::Size input_geometry_;
	int num_channels_;
};