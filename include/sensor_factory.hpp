#pragma once

#include "sensor_interface.hpp"
#include <string>
#include <unordered_map>

namespace hal {

// ─────────────────────────────────────────────────────────────────────────────
//  SensorConfig — flat key/value bag (could be backed by JSON/TOML/env-vars)
// ─────────────────────────────────────────────────────────────────────────────
struct SensorConfig {
    /// "simulated" | "hardware"
    std::string mode{"simulated"};

    /// Sensor kind: "gps" | "video" | "imu"
    std::string kind{"gps"};

    // Simulated-mode options
    std::string sim_file{};         ///< Empty → use synthetic generator
    uint32_t    sim_num_samples{100};

    // Hardware-mode options
    std::string hw_device{};        ///< e.g. "/dev/ttyUSB0"
    int         hw_baud{9600};
    uint32_t    hw_width{640};
    uint32_t    hw_height{480};
    float       hw_fps{30.f};

    // Shared
    float rate_hz{10.f};

    // ── Parse from a simple key=value text file ──────────────────────────────
    static SensorConfig from_file(const std::string& path);

    // ── Parse from a flat string map (e.g. CLI args, environment) ───────────
    static SensorConfig from_map(const std::unordered_map<std::string,std::string>& m);

    // ── Dump current config to stdout ────────────────────────────────────────
    void print() const;
};

// ─────────────────────────────────────────────────────────────────────────────
//  SensorFactory — single point of construction for the inference pipeline
// ─────────────────────────────────────────────────────────────────────────────
class SensorFactory {
public:
    /// Build the correct ISensorInput implementation from a SensorConfig.
    /// Throws std::invalid_argument for unknown mode/kind combinations.
    static SensorPtr create(const SensorConfig& cfg);
};

} // namespace hal
