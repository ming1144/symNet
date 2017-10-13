#include "extracter.h"


using namespace caffe;

Extracter::Extracter(const string& model_file,
	const string& trained_file)
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
	//std::cout << model_file << std::endl << trained_file << std::endl;
	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();

	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Extracter::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Extracter::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	//cv::imshow("test", sample_resized);
	//cv::waitKey(0);



	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat temp1(sample_float.rows, sample_float.cols, CV_32FC1);



	temp1 = cv::Scalar(128);

	cv::subtract(sample_float, temp1, sample_float);

	cv::Mat temp2(sample_float.rows, sample_float.cols, CV_32FC1);

	temp2 = cv::Scalar(160);

	cv::divide(sample_float, temp2, sample_float);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_float, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

void Extracter::featureExtract(const string& input_file)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	cv::Mat img = cv::imread(input_file);

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Blob<float>* output_layer = net_->output_blobs()[0];

	Preprocess(img, &input_channels);

	net_->Forward();
}

void Extracter::featureExtract(const cv::Mat& img)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();


	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Blob<float>* output_layer = net_->output_blobs()[0];

	Preprocess(img, &input_channels);

	net_->Forward();

	const float* p_output = output_layer->cpu_data();
}

void Extracter::featureExtract(const cv::Mat& img, float* features)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();


	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Blob<float>* output_layer = net_->output_blobs()[0];

	Preprocess(img, &input_channels);

	net_->Forward();

	const float* p_output = output_layer->cpu_data();

	for (int i = 0; i < 64 * 64; i++, p_output++, features++)
	{
		*features = *p_output;
	}
}