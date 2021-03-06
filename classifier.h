#include <iostream>
#include <string>
#include <vector>

#include <caffe\caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::string;
using std::vector;

typedef std::pair<string, float> Prediction;

using boost::shared_ptr;
using caffe::Net;

class Classifier {
public:
	Classifier();
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	Classifier(const string& model_file,
		const string& trained_file,
		const string& label_file);

	Classifier(const string& model_file,
		const string& trained_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

	float featureCompare(float* feature1, float* feature2);
private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;

	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	bool useMean;
	std::vector<string> labels_;
	bool useLabel;
};