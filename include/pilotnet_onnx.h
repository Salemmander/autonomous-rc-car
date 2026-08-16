#pragma once

#include <cstdint>
#include <memory>
#include <vector>

class PilotNetOnnx {
public:
    PilotNetOnnx(const char* model_path, int height, int width);
    ~PilotNetOnnx();

    PilotNetOnnx(const PilotNetOnnx&) = delete;
    PilotNetOnnx& operator=(const PilotNetOnnx&) = delete;

    // hwc_rgb: H*W*3 uint8. Returns false if size is wrong.
    bool infer(const std::vector<std::uint8_t>& hwc_rgb, float& steering, float& throttle);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
