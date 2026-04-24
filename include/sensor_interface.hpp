#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace hal {

// ─────────────────────────────────────────────
//  Metadata attached to every sample/frame
// ─────────────────────────────────────────────
struct SensorMetadata {
    uint64_t    timestamp_us;   ///< Microseconds since epoch (or boot)
    uint32_t    sequence_id;    ///< Monotonically increasing sample counter
    std::string sensor_id;      ///< Logical name of the source
    bool        is_simulated;   ///< True when data comes from a simulated source
};

// ─────────────────────────────────────────────
//  Typed payloads  (extend as needed)
// ─────────────────────────────────────────────
struct GpsSample {
    double latitude_deg;
    double longitude_deg;
    double altitude_m;
    float  speed_mps;
    float  heading_deg;
};

struct VideoFrame {
    uint32_t             width;
    uint32_t             height;
    uint32_t             channels;          ///< e.g. 1 = mono, 3 = RGB
    std::vector<uint8_t> pixels;            ///< Row-major, interleaved
};

struct ImuSample {
    float accel_x, accel_y, accel_z;       ///< m/s²
    float gyro_x,  gyro_y,  gyro_z;        ///< rad/s
    float mag_x,   mag_y,   mag_z;         ///< µT
};

/// Generic payload — add new sensor types here without touching the interface
using SensorPayload = std::variant<GpsSample, VideoFrame, ImuSample>;

// ─────────────────────────────────────────────
//  A single reading coming out of any sensor
// ─────────────────────────────────────────────
struct SensorSample {
    SensorMetadata metadata;
    SensorPayload  payload;
};

// ─────────────────────────────────────────────
//  Abstract sensor interface  (the HAL contract)
// ─────────────────────────────────────────────
class ISensorInput {
public:
    virtual ~ISensorInput() = default;

    /// Open / initialise the underlying source.
    /// Returns true on success.
    virtual bool open() = 0;

    /// Release any resources (file handles, device fds, sockets…).
    virtual void close() = 0;

    /// Pull the next sample.
    /// Returns std::nullopt when the source is exhausted or in error.
    virtual std::optional<SensorSample> next_sample() = 0;

    /// True while the source can still produce samples.
    virtual bool is_ready() const = 0;

    /// Human-readable description of this source.
    virtual std::string description() const = 0;

    // ── Optional capability queries ──────────────────────────────────────
    /// Nominal output rate (Hz).  0 = unknown / as-fast-as-possible.
    virtual float nominal_rate_hz() const { return 0.f; }

    /// Seek to the beginning (only meaningful for file-backed sources).
    virtual bool rewind() { return false; }
};

// ─────────────────────────────────────────────
//  Convenience alias
// ─────────────────────────────────────────────
using SensorPtr = std::unique_ptr<ISensorInput>;

} // namespace hal
