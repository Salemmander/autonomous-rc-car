#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

class MjpegStream {
public:
    explicit MjpegStream(int port = 8080);
    ~MjpegStream();

    MjpegStream(const MjpegStream&) = delete;
    MjpegStream& operator=(const MjpegStream&) = delete;

    bool start();
    void stop();

    // RGB888 packed H*W*3. Replaces the frame sent to browsers.
    void send(const std::vector<std::uint8_t>& rgb, int width, int height);

private:
    void loop();

    int port_;
    int listen_fd_{-1};
    std::atomic<bool> running_{false};
    std::thread thread_;

    std::mutex jpeg_mu_;
    std::condition_variable jpeg_cv_;
    std::vector<std::uint8_t> jpeg_;
    uint64_t jpeg_gen_{0};
};
