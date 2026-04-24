#pragma once

#include "sensor_interface.hpp"

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace hal {

// ─────────────────────────────────────────────────────────────────────────────
//  SimulatedSensor
//  Supports two modes:
//    1. Synthetic  — a user-supplied generator lambda produces each sample.
//    2. File-based — a pre-recorded log (CSV or binary) is replayed in order.
//
//  Both modes share the same ISensorInput contract; the inference pipeline
//  is entirely unaware of which is active.
// ─────────────────────────────────────────────────────────────────────────────
class SimulatedSensor final : public ISensorInput {
public:
    // ── Generator-based construction ────────────────────────────────────────
    /// @param generator   Callable that produces one SensorPayload per call.
    ///                    Return std::nullopt to signal end-of-stream.
    /// @param sensor_id   Logical name reported in metadata.
    /// @param rate_hz     Nominal playback rate (informational only here).
    /// @param max_samples 0 = unlimited.
    explicit SimulatedSensor(
        std::function<std::optional<SensorPayload>()> generator,
        std::string  sensor_id  = "simulated",
        float        rate_hz    = 10.f,
        uint32_t     max_samples = 0);

    // ── File-based construction ──────────────────────────────────────────────
    /// @param file_path   Path to a pre-recorded log.
    /// @param parser      Callable that converts one raw text line → payload.
    ///                    Return std::nullopt to skip a line (e.g. comments).
    /// @param sensor_id   Logical name reported in metadata.
    /// @param rate_hz     Nominal playback rate.
    explicit SimulatedSensor(
        std::string  file_path,
        std::function<std::optional<SensorPayload>(const std::string&)> parser,
        std::string  sensor_id = "file-simulated",
        float        rate_hz   = 10.f);

    // ── ISensorInput ─────────────────────────────────────────────────────────
    bool open()   override;
    void close()  override;
    std::optional<SensorSample> next_sample() override;
    bool        is_ready()       const override;
    std::string description()    const override;
    float       nominal_rate_hz() const override { return rate_hz_; }
    bool        rewind()         override;

private:
    enum class Mode { Generator, FileBased };
    Mode mode_;

    // Generator mode
    std::function<std::optional<SensorPayload>()> generator_;

    // File-based mode
    std::string file_path_;
    std::function<std::optional<SensorPayload>(const std::string&)> parser_;
    std::vector<std::string> lines_;
    std::size_t              line_cursor_{0};

    // Common
    std::string sensor_id_;
    float       rate_hz_;
    uint32_t    max_samples_;
    uint32_t    samples_produced_{0};
    bool        ready_{false};

    uint64_t now_us() const;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Factory helpers — ready-made synthetic sources
// ─────────────────────────────────────────────────────────────────────────────

/// Circular GPS track around a centre point.
SensorPtr make_synthetic_gps(uint32_t num_samples = 100,
                             double   centre_lat  = 37.7749,
                             double   centre_lon  = -122.4194);

/// Checkerboard video frames (RGB, configurable resolution).
SensorPtr make_synthetic_video(uint32_t width       = 320,
                               uint32_t height      = 240,
                               uint32_t num_frames  = 30,
                               float    rate_hz     = 30.f);

/// Sinusoidal IMU data (simulates vibration on all axes).
SensorPtr make_synthetic_imu(uint32_t num_samples = 200, float rate_hz = 100.f);

/// File-backed GPS replayer (expects CSV: lat,lon,alt,speed,heading).
SensorPtr make_file_gps(const std::string& csv_path, float rate_hz = 10.f);

} // namespace hal
