#include "pilotnet_onnx.h"

#include <onnxruntime_cxx_api.h>

struct PilotNetOnnx::Impl {
    int height;
    int width;
    size_t plane;
    std::vector<float> input;
    int64_t shape[4];
    const char* in_names[1] = {"rgb_nchw_0_255"};
    const char* out_names[1] = {"steer_throttle"};

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "pilotnet"};
    Ort::SessionOptions opts{};
    Ort::Session session;
    Ort::MemoryInfo mem;

    Impl(const char* model_path, int h, int w)
        : height(h),
          width(w),
          plane(static_cast<size_t>(h) * static_cast<size_t>(w)),
          input(3 * plane),
          shape{1, 3, h, w},
          session(env, model_path, opts),
          mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};

PilotNetOnnx::PilotNetOnnx(const char* model_path, int height, int width)
    : impl_(std::make_unique<Impl>(model_path, height, width)) {}

PilotNetOnnx::~PilotNetOnnx() = default;

bool PilotNetOnnx::infer(const std::vector<std::uint8_t>& hwc_rgb, float& steering,
                         float& throttle) {
    if (hwc_rgb.size() != 3 * impl_->plane) {
        return false;
    }

    const int h = impl_->height;
    const int w = impl_->width;
    const size_t plane = impl_->plane;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y * w + x);
            impl_->input[0 * plane + i] = hwc_rgb[i * 3 + 0];
            impl_->input[1 * plane + i] = hwc_rgb[i * 3 + 1];
            impl_->input[2 * plane + i] = hwc_rgb[i * 3 + 2];
        }
    }

    auto tensor = Ort::Value::CreateTensor<float>(impl_->mem, impl_->input.data(),
                                                  impl_->input.size(), impl_->shape, 4);
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, impl_->in_names, &tensor, 1,
                                      impl_->out_names, 1);
    const float* y = outputs[0].GetTensorMutableData<float>();
    steering = y[0];
    throttle = y[1];
    return true;
}
