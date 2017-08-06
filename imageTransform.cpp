#include "imageTransform.h"

void imageTransform::localBinaryPattern(const cv::Mat *input, cv::Mat *output)
{
	int i, j;
	for (i = 1; i < input->rows - 1; i++)
	{
		for (j = 1; j < input->cols - 1; j++)
		{
			uchar temp = input->at<uchar>(i, j);
			uchar num = 0;
			if (temp > input->at<uchar>(i - 1, j - 1)) num = num | 1;
			if (temp > input->at<uchar>(i, j - 1)) num = num | 2;
			if (temp > input->at<uchar>(i + 1, j - 1)) num = num | 4;
			if (temp > input->at<uchar>(i + 1, j)) num = num | 8;
			if (temp > input->at<uchar>(i + 1, j + 1)) num = num | 16;
			if (temp > input->at<uchar>(i, j + 1)) num = num | 32;
			if (temp > input->at<uchar>(i - 1, j + 1)) num = num | 64;
			if (temp > input->at<uchar>(i - 1, j)) num = num | 128;

			output->at<uchar>(i, j) = num;
		}
	}
}

void imageTransform::nakagami(const cv::Mat* input, cv::Mat*output, int maskSize)
{
	int i, j, x, y, mean_element;
	double mean, variance;
	cv::Vec3b temp;
	uchar tempUchar;
	cv::Mat tempImg(input->rows + maskSize - 1, input->cols + maskSize - 1, input->type(), cv::Scalar(0));
	input->copyTo(tempImg.rowRange(maskSize / 2, maskSize / 2 + input->rows).colRange(maskSize / 2, maskSize / 2 + input->cols));

	for (i = maskSize / 2; i < maskSize / 2 + input->rows; i++)
	{
		for (j = maskSize / 2; j < maskSize / 2 + input->cols; j++)
		{
			mean = 0;
			mean_element = 0;
			for (y = i - maskSize / 2; y <= i + maskSize / 2; y++)
			{
				for (x = j - maskSize / 2; x <= j + maskSize / 2; x++)
				{
					temp = tempImg.at<cv::Vec3b>(y, x);
					if (temp.val[0] == temp.val[1] && temp.val[0] == temp.val[2] && temp.val[0] != 0 && temp.val[1] != 0 && temp.val[2] != 0)
					{
						tempUchar = (temp.val[0] + temp.val[1] + temp.val[2]) / 3;
						mean += tempUchar * tempUchar;
						mean_element++;
					}
				}
			}
			if (mean == 0 || mean_element == 0)
			{
				output->at<uchar>(i - maskSize / 2, j - maskSize / 2) = 0;
			}

			mean /= (double)mean_element;
			variance = 0;
			for (y = i - maskSize / 2; y <= i + maskSize / 2; y++)
			{
				for (x = j - maskSize / 2; x <= j + maskSize / 2; x++)
				{
					temp = tempImg.at<cv::Vec3b>(y, x);
					if (temp.val[0] == temp.val[1] && temp.val[0] == temp.val[2] && temp.val[0] != 0 && temp.val[1] != 0 && temp.val[2] != 0)
					{
						tempUchar = (temp.val[0] + temp.val[1] + temp.val[2]) / 3;
						variance += (tempUchar * tempUchar - mean) * (tempUchar * tempUchar - mean);
					}
				}
			}
			variance /= (double)mean_element;

			if (variance == 0)
			{
				output->at<uchar>(i - maskSize / 2, j - maskSize / 2) = 0;
			}
			output->at<uchar>(i - maskSize / 2, j - maskSize / 2) = mean * mean / variance;
		}
	}
}
