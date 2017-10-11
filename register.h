#include <caffe\common.hpp>
#include <caffe\layers\input_layer.hpp>
#include <caffe\layers\inner_product_layer.hpp>
#include <caffe\layers\flatten_layer.hpp>
#include <caffe\layers\concat_layer.hpp>
#include <caffe\layers\dummy_data_layer.hpp>
#include <caffe\layers\crop_layer.hpp>
#include <caffe\layers\dropout_layer.hpp>

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(FlattenLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	extern INSTANTIATE_CLASS(DummyDataLayer);
	extern INSTANTIATE_CLASS(CropLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
}