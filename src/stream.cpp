#include "stream.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>

MjpegStream::MjpegStream(int port) : port_(port) {}

MjpegStream::~MjpegStream() {
    stop();
}

void MjpegStream::send(const std::vector<std::uint8_t>& rgb, int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const size_t need = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    if (rgb.size() != need) {
        return;
    }

    cv::Mat rgb_mat(height, width, CV_8UC3, const_cast<std::uint8_t*>(rgb.data()));
    cv::Mat bgr;
    cv::cvtColor(rgb_mat, bgr, cv::COLOR_RGB2BGR);
    std::vector<std::uint8_t> encoded;
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 60};
    if (!cv::imencode(".jpg", bgr, encoded, params) || encoded.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(jpeg_mu_);
        jpeg_ = std::move(encoded);
        ++jpeg_gen_;
    }
    jpeg_cv_.notify_one();
}

bool MjpegStream::start() {
    if (running_ || thread_.joinable()) {
        return true;
    }

    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        perror("socket");
        return false;
    }

    int yes = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    if (listen(listen_fd_, 8) < 0) {
        perror("listen");
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    running_ = true;
    thread_ = std::thread([this] { loop(); });
    return true;
}

void MjpegStream::stop() {
    running_ = false;
    jpeg_cv_.notify_all();
    if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
    }
}

void MjpegStream::loop() {
    while (running_) {
        int client = accept(listen_fd_, nullptr, nullptr);
        if (client < 0) {
            if (!running_) {
                break;
            }
            perror("accept");
            break;
        }

        int nodelay = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

        char buf[1024];
        if (read(client, buf, sizeof(buf) - 1) < 0) {
            perror("read");
            close(client);
            continue;
        }

        const char* stream_hdr =
            "HTTP/1.1 200 OK\r\n"
            "Cache-Control: no-cache, no-store, must-revalidate\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "\r\n";
        if (write(client, stream_hdr, strlen(stream_hdr)) < 0) {
            perror("write stream hdr");
            close(client);
            continue;
        }

        uint64_t last_gen = 0;
        while (running_) {
            std::vector<std::uint8_t> jpeg;
            {
                std::unique_lock<std::mutex> lock(jpeg_mu_);
                jpeg_cv_.wait_for(lock, std::chrono::milliseconds(200), [&] {
                    return !running_ || jpeg_gen_ != last_gen;
                });
                if (!running_) {
                    break;
                }
                if (jpeg_gen_ == last_gen || jpeg_.empty()) {
                    continue;
                }
                last_gen = jpeg_gen_;
                jpeg = jpeg_;
            }

            char part_hdr[128];
            int hdr_len = std::snprintf(
                part_hdr, sizeof(part_hdr),
                "--frame\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: %zu\r\n"
                "\r\n",
                jpeg.size());

            if (write(client, part_hdr, hdr_len) < 0 ||
                write(client, jpeg.data(), jpeg.size()) < 0 ||
                write(client, "\r\n", 2) < 0) {
                break;
            }
        }
        close(client);
    }
}
