#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp> 

class imageTransform{
public:
	static void localBinaryPattern(const cv::Mat *input, cv::Mat* output);

	static void nakagami(const cv::Mat* input, cv::Mat*output, int maskSize = 7);
};