#include "chrono"
#include "iostream"
#include "rknn_api.h"
#include "opencv2/opencv.hpp"

typedef struct {
	size_t size;
	unsigned char *content;
}FileContent;

void load_file(const char *filename, FileContent &content)
{
	// 文件锁

	FILE *fp = fopen(filename, "rb");
	if (fp == nullptr) {
		printf("fopen %s fail!\n", filename);
		return;
	}
	fseek(fp, 0, SEEK_END);
	int model_len = ftell(fp);
	content.content = (unsigned char*)malloc(model_len);
	fseek(fp, 0, SEEK_SET);
	if (model_len != fread(content.content, 1, model_len, fp)) {
		printf("fread %s fail!\n", filename);
		free(content.content);
		return;
	}
	content.size = model_len;
	if (fp) {
		fclose(fp);
	}
}


inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

int main(int argc, char const *argv[])
{
	rknn_context ctx;
	int model_size[2];   // width height
	rknn_input_output_num io_num;
	rknn_tensor_attr *output_attrs;
	rknn_output *outputs_tensor;

    int ret;
    FileContent content;
    load_file("./u2netp_simple.rknn", content);
    ret = rknn_init(&(ctx), (unsigned char*)content.content, content.size, 0, NULL);
    rknn_core_mask mode = RKNN_NPU_CORE_AUTO;
    mode = RKNN_NPU_CORE_AUTO;
    ret = rknn_set_core_mask(ctx, mode);
    if(ret != RKNN_SUCC)
    {
        printf("[ Inference create error: rknn_init ]");
    }

	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret != RKNN_SUCC)
	{
		printf("[ Inference create error: rknn_query ]");
	}

	rknn_tensor_attr input_attrs;
	input_attrs.index = 0;
	ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs, sizeof(rknn_tensor_attr));

	// 打印输入shape
    std::cout << "[ W Model Shape: " << input_attrs.dims[3] << " " << input_attrs.dims[1] << " " <<input_attrs.dims[2] << " ]" << std::endl;

	output_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr)*(io_num.n_output));
	memset(output_attrs, 0, sizeof(rknn_tensor_attr)*(io_num.n_output));

	outputs_tensor = (rknn_output*)malloc(sizeof(rknn_output)*(io_num.n_output));
	for (int i = 0; i < io_num.n_output; i++)
	{
		output_attrs[i].index = i;
		ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));

		// 手动指定输出空间
		(outputs_tensor)[i].want_float = 1;
		(outputs_tensor)[i].is_prealloc = 1;
		(outputs_tensor)[i].buf = malloc(sizeof(float) * output_attrs[i].size);
		if (ret != RKNN_SUCC)
		{
			printf("[ Inference create error: rknn_query ]");
		}
		// 打印输出shape
			std::cout << "Model OutShape " << i << ": ";
			for(size_t index=0; index < output_attrs[i].n_dims; index ++)
			{
				std::cout << output_attrs[i].dims[index] << " ";
			}
			std::cout << std::endl;
	}

	// 预处理
	cv::Mat img = cv::imread("./10002-01.jpg");
	// 
	ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs, sizeof(rknn_tensor_attr));
	rknn_input inputs[io_num.n_input];
	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].size = 320 * 320 * 3;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	inputs[0].buf = img.data;
	
	ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
	if (ret != RKNN_SUCC)
	{
		printf("[ Inference forward error: rknn_inputs_set ]");
	}

    // while (1) 
    {
		auto start = std::chrono::high_resolution_clock::now();
		ret = rknn_run(ctx, nullptr);
		if (ret != RKNN_SUCC)
		{
			printf("[ Inference forward error: rknn_run ]");
		}
		ret = rknn_outputs_get(ctx, io_num.n_output, outputs_tensor, NULL);
		if (ret != RKNN_SUCC)
		{
			printf("[ Inference forward error: rknn_outputs_get ]");
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "forwart time: " << duration.count() * 1.0 /1000 << " ms" << std::endl;
	}

	float *output = (float *)outputs_tensor->buf;
	char tmp[320*320];
	for (int i=0; i< 320*320; i++) {
		tmp[i] = sigmoid(output[i]) * 255;
	}
	cv::Mat res_img(320, 320, CV_8UC1, tmp);
	cv::imwrite("res.jpg", res_img);
    return 0;
}
