#include <caffe\common.hpp>
#include <caffe\layers\input_layer.hpp>
#include <caffe\layers\inner_product_layer.hpp>
#include <caffe\layers\concat_layer.hpp>
#include <caffe\layers\dropout_layer.hpp>
#include <caffe\layers\mirror_layer.hpp>
#include <caffe\layers\gradient_layer.hpp>

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(MirrorLayer);
	extern INSTANTIATE_CLASS(GradientLayer);
}