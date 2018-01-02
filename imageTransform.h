#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp> 

#define M_PI 3.14159

class imageTransform{
public:
	static cv::Mat localBinaryPattern(const cv::Mat& input);

	static void nakagami(const cv::Mat* input, cv::Mat*output, int maskSize = 7);

	static cv::Mat imageRotate(const cv::Mat& img, float angle);

	static void gradientAndAngle(cv::Mat& img, cv::Mat& gradient, cv::Mat& angle);
};