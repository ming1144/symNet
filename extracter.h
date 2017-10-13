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
		const string& trained_file);

	void featureExtract(const string& input_file);
	void featureExtract(const cv::Mat& img);
	void featureExtract(const cv::Mat& img, float* features);

private:
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;

	cv::Size input_geometry_;
	int num_channels_;
};