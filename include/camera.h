#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <mutex>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

class CameraController {
private:
    const char* bus_path = "/dev/video0";

    int fd{-1};
    int width, height;
    unsigned int image_size{0};

    struct CamBuffer {
        void* start{};
        std::size_t length{0};
    };
    std::vector<CamBuffer> buffers;

    // Latest MJPEG frame copy (JPEG bytes; decode to RGB later).
    std::vector<std::uint8_t> latest_frame_;
    bool has_frame_{false};
    mutable std::mutex frame_mutex_;

    void release() {
        if (fd < 0) {
            return;
        }

        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd, VIDIOC_STREAMOFF, &type);

        for (auto& b : buffers) {
            if (b.start && b.start != MAP_FAILED) {
                munmap(b.start, b.length);
            }
            b.start = nullptr;
            b.length = 0;
        }
        buffers.clear();

        close(fd);
        fd = -1;
    }

public:
    CameraController(int width, int height) : width(width), height(height) {
        fd = open(bus_path, O_RDWR);
        if (fd < 0) {
            perror("open /dev/video0");
            return;
        }

        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;

        if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
            perror("ioctl VIDIOC_S_FMT");
            release();
            return;
        }

        // Driver may adjust size; use what was actually accepted.
        if (static_cast<int>(fmt.fmt.pix.width) != width ||
            static_cast<int>(fmt.fmt.pix.height) != height) {
            std::cerr << "camera: requested " << width << "x" << height << ", driver gave "
                      << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << "\n";
        }
        this->width = fmt.fmt.pix.width;
        this->height = fmt.fmt.pix.height;
        image_size = fmt.fmt.pix.sizeimage;

        v4l2_requestbuffers req{};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
            perror("ioctl VIDIOC_REQBUFS");
            release();
            return;
        }

        buffers.resize(req.count);
        for (unsigned int i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;

            if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
                perror("ioctl VIDIOC_QUERYBUF");
                release();
                return;
            }

            void* start =
                mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
            if (start == MAP_FAILED) {
                perror("mmap");
                release();
                return;
            }

            buffers[i].start = start;
            buffers[i].length = buf.length;

            if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
                perror("ioctl VIDIOC_QBUF");
                release();
                return;
            }
        }

        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
            perror("ioctl VIDIOC_STREAMON");
            release();
            return;
        }
    }

    ~CameraController() {
        release();
    }

    bool ok() const {
        return fd >= 0;
    }

    int getWidth() const {
        return width;
    }

    int getHeight() const {
        return height;
    }

    bool hasFrame() const {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        return has_frame_;
    }

    // Returns a copy of the latest MJPEG frame (JPEG bytes), or empty if none yet.
    std::vector<std::uint8_t> getFrame() const {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        return latest_frame_;
    }

    void capture() {
        if (fd < 0) {
            return;
        }

        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
            perror("ioctl VIDIOC_DQBUF");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_.resize(buf.bytesused);
            std::memcpy(latest_frame_.data(), buffers[buf.index].start, buf.bytesused);
            has_frame_ = true;
        }

        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("ioctl VIDIOC_QBUF");
            return;
        }
    }
};
