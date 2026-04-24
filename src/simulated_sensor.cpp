#include "simulated_sensor.hpp"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hal {

// ─────────────────────────────────────────────────────────────────────────────
//  Constructors
// ─────────────────────────────────────────────────────────────────────────────
SimulatedSensor::SimulatedSensor(
    std::function<std::optional<SensorPayload>()> generator,
    std::string  sensor_id,
    float        rate_hz,
    uint32_t     max_samples)
    : mode_(Mode::Generator)
    , generator_(std::move(generator))
    , sensor_id_(std::move(sensor_id))
    , rate_hz_(rate_hz)
    , max_samples_(max_samples)
{}

SimulatedSensor::SimulatedSensor(
    std::string  file_path,
    std::function<std::optional<SensorPayload>(const std::string&)> parser,
    std::string  sensor_id,
    float        rate_hz)
    : mode_(Mode::FileBased)
    , file_path_(std::move(file_path))
    , parser_(std::move(parser))
    , sensor_id_(std::move(sensor_id))
    , rate_hz_(rate_hz)
    , max_samples_(0)
{}

// ─────────────────────────────────────────────────────────────────────────────
//  ISensorInput implementation
// ─────────────────────────────────────────────────────────────────────────────
bool SimulatedSensor::open() {
    samples_produced_ = 0;
    line_cursor_      = 0;
    lines_.clear();

    if (mode_ == Mode::FileBased) {
        std::ifstream ifs(file_path_);
        if (!ifs) {
            return false;   // caller should log the error
        }
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty()) lines_.push_back(std::move(line));
        }
        if (lines_.empty()) return false;
    }

    ready_ = true;
    return true;
}

void SimulatedSensor::close() {
    ready_ = false;
    lines_.clear();
}

std::optional<SensorSample> SimulatedSensor::next_sample() {
    if (!ready_) return std::nullopt;
    if (max_samples_ > 0 && samples_produced_ >= max_samples_) {
        ready_ = false;
        return std::nullopt;
    }

    std::optional<SensorPayload> payload;

    if (mode_ == Mode::Generator) {
        payload = generator_();
    } else {
        // Advance past lines that the parser skips (e.g., comment lines)
        while (line_cursor_ < lines_.size()) {
            payload = parser_(lines_[line_cursor_++]);
            if (payload) break;
        }
        if (!payload) {
            ready_ = false;
            return std::nullopt;
        }
    }

    if (!payload) {
        ready_ = false;
        return std::nullopt;
    }

    SensorSample sample;
    sample.metadata.timestamp_us = now_us();
    sample.metadata.sequence_id  = samples_produced_++;
    sample.metadata.sensor_id    = sensor_id_;
    sample.metadata.is_simulated = true;
    sample.payload               = std::move(*payload);
    return sample;
}

bool SimulatedSensor::is_ready() const { return ready_; }

std::string SimulatedSensor::description() const {
    std::ostringstream oss;
    oss << "[SimulatedSensor id=" << sensor_id_;
    if (mode_ == Mode::FileBased) oss << " file=" << file_path_;
    else                          oss << " mode=synthetic";
    oss << " rate=" << rate_hz_ << "Hz]";
    return oss.str();
}

bool SimulatedSensor::rewind() {
    if (mode_ == Mode::FileBased) {
        line_cursor_      = 0;
        samples_produced_ = 0;
        ready_            = !lines_.empty();
        return ready_;
    }
    // Generator mode: reset counter (generator itself is stateless in most cases)
    samples_produced_ = 0;
    ready_            = true;
    return true;
}

uint64_t SimulatedSensor::now_us() const {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<microseconds>(
            system_clock::now().time_since_epoch()).count());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Factory helpers
// ─────────────────────────────────────────────────────────────────────────────
SensorPtr make_synthetic_gps(uint32_t num_samples,
                             double   centre_lat,
                             double   centre_lon) {
    const double radius_deg = 0.001;   // ~111 m
    auto counter = std::make_shared<uint32_t>(0);

    auto gen = [=]() -> std::optional<SensorPayload> {
        if (*counter >= num_samples) return std::nullopt;
        double angle = 2.0 * M_PI * (*counter) / num_samples;
        GpsSample s;
        s.latitude_deg  = centre_lat + radius_deg * std::cos(angle);
        s.longitude_deg = centre_lon + radius_deg * std::sin(angle);
        s.altitude_m    = 10.0 + std::sin(angle) * 5.0;
        s.speed_mps     = 5.0f + static_cast<float>(std::sin(angle * 3)) * 1.5f;
        s.heading_deg   = static_cast<float>(std::fmod(angle * 180.0 / M_PI + 360.0, 360.0));
        ++(*counter);
        return s;
    };
    return std::make_unique<SimulatedSensor>(gen, "gps-synthetic", 10.f, num_samples);
}

SensorPtr make_synthetic_video(uint32_t width,
                               uint32_t height,
                               uint32_t num_frames,
                               float    rate_hz) {
    auto frame_idx = std::make_shared<uint32_t>(0);
    const uint32_t tile = 40;

    auto gen = [=]() -> std::optional<SensorPayload> {
        if (*frame_idx >= num_frames) return std::nullopt;
        VideoFrame f;
        f.width    = width;
        f.height   = height;
        f.channels = 3;
        f.pixels.resize(width * height * 3);
        // Animated checkerboard: tile phase shifts each frame
        uint32_t phase = *frame_idx * 4;
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                bool white = (((x + phase) / tile) + (y / tile)) % 2 == 0;
                uint8_t v  = white ? 220 : 35;
                uint32_t idx = (y * width + x) * 3;
                f.pixels[idx+0] = v;
                f.pixels[idx+1] = white ? static_cast<uint8_t>(v - (*frame_idx % 40)) : v;
                f.pixels[idx+2] = white ? v : static_cast<uint8_t>(v + (*frame_idx % 60));
            }
        }
        ++(*frame_idx);
        return f;
    };
    return std::make_unique<SimulatedSensor>(gen, "camera-synthetic", rate_hz, num_frames);
}

SensorPtr make_synthetic_imu(uint32_t num_samples, float rate_hz) {
    auto counter = std::make_shared<uint32_t>(0);
    auto gen = [=]() -> std::optional<SensorPayload> {
        if (*counter >= num_samples) return std::nullopt;
        double t = static_cast<double>(*counter) / rate_hz;
        ImuSample s;
        s.accel_x =  0.1f * static_cast<float>(std::sin(2 * M_PI * 1.0 * t));
        s.accel_y =  0.05f* static_cast<float>(std::cos(2 * M_PI * 2.0 * t));
        s.accel_z =  9.81f + 0.02f * static_cast<float>(std::sin(2 * M_PI * 0.5 * t));
        s.gyro_x  =  0.01f * static_cast<float>(std::cos(2 * M_PI * 3.0 * t));
        s.gyro_y  = -0.01f * static_cast<float>(std::sin(2 * M_PI * 3.0 * t));
        s.gyro_z  =  0.005f* static_cast<float>(std::sin(2 * M_PI * 0.8 * t));
        s.mag_x   = 20.f;  s.mag_y = 5.f;  s.mag_z = -42.f;
        ++(*counter);
        return s;
    };
    return std::make_unique<SimulatedSensor>(gen, "imu-synthetic", rate_hz, num_samples);
}

SensorPtr make_file_gps(const std::string& csv_path, float rate_hz) {
    auto parser = [](const std::string& line) -> std::optional<SensorPayload> {
        if (line.empty() || line[0] == '#') return std::nullopt;
        std::istringstream ss(line);
        std::string token;
        std::vector<double> fields;
        while (std::getline(ss, token, ',')) {
            try { fields.push_back(std::stod(token)); }
            catch (...) { return std::nullopt; }
        }
        if (fields.size() < 5) return std::nullopt;
        GpsSample s;
        s.latitude_deg  = fields[0];
        s.longitude_deg = fields[1];
        s.altitude_m    = fields[2];
        s.speed_mps     = static_cast<float>(fields[3]);
        s.heading_deg   = static_cast<float>(fields[4]);
        return s;
    };
    return std::make_unique<SimulatedSensor>(csv_path, parser, "gps-file", rate_hz);
}

} // namespace hal
