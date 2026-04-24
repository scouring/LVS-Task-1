#include "sensor_factory.hpp"
#include "simulated_sensor.hpp"
#include "hardware_sensor.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace hal {

// ─────────────────────────────────────────────────────────────────────────────
//  SensorConfig helpers
// ─────────────────────────────────────────────────────────────────────────────
SensorConfig SensorConfig::from_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open config file: " + path);

    std::unordered_map<std::string, std::string> m;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key   = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        // trim whitespace
        auto trim = [](std::string& s) {
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
            while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t' ||
                                  s.back()  == '\r'|| s.back()  == '\n')) s.pop_back();
        };
        trim(key); trim(value);
        m[key] = value;
    }
    return from_map(m);
}

SensorConfig SensorConfig::from_map(
    const std::unordered_map<std::string, std::string>& m) {
    SensorConfig cfg;
    auto get = [&](const std::string& k, const std::string& def) -> std::string {
        auto it = m.find(k);
        return it != m.end() ? it->second : def;
    };
    cfg.mode            = get("mode",            "simulated");
    cfg.kind            = get("kind",            "gps");
    cfg.sim_file        = get("sim_file",        "");
    cfg.hw_device       = get("hw_device",       "");
    cfg.sim_num_samples = static_cast<uint32_t>(std::stoul(get("sim_num_samples", "100")));
    cfg.hw_baud         = std::stoi(get("hw_baud",   "9600"));
    cfg.hw_width        = static_cast<uint32_t>(std::stoul(get("hw_width",  "640")));
    cfg.hw_height       = static_cast<uint32_t>(std::stoul(get("hw_height", "480")));
    cfg.hw_fps          = std::stof(get("hw_fps",  "30"));
    cfg.rate_hz         = std::stof(get("rate_hz", "10"));
    return cfg;
}

void SensorConfig::print() const {
    std::cout << "┌─ SensorConfig ──────────────────────────\n"
              << "│  mode            = " << mode            << "\n"
              << "│  kind            = " << kind            << "\n"
              << "│  rate_hz         = " << rate_hz         << "\n";
    if (mode == "simulated") {
        std::cout
              << "│  sim_file        = " << (sim_file.empty() ? "(synthetic)" : sim_file) << "\n"
              << "│  sim_num_samples = " << sim_num_samples  << "\n";
    } else {
        std::cout
              << "│  hw_device       = " << hw_device  << "\n"
              << "│  hw_baud         = " << hw_baud    << "\n"
              << "│  hw_width        = " << hw_width   << "\n"
              << "│  hw_height       = " << hw_height  << "\n"
              << "│  hw_fps          = " << hw_fps     << "\n";
    }
    std::cout << "└─────────────────────────────────────────\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  SensorFactory::create
// ─────────────────────────────────────────────────────────────────────────────
SensorPtr SensorFactory::create(const SensorConfig& cfg) {
    if (cfg.mode == "simulated") {
        // ── File-backed simulated source ─────────────────────────────────────
        if (!cfg.sim_file.empty()) {
            if (cfg.kind == "gps") return make_file_gps(cfg.sim_file, cfg.rate_hz);
            throw std::invalid_argument(
                "File-based simulation only supported for kind=gps (got: " + cfg.kind + ")");
        }

        // ── Synthetic (generator) sources ────────────────────────────────────
        if (cfg.kind == "gps")   return make_synthetic_gps(cfg.sim_num_samples);
        if (cfg.kind == "video") return make_synthetic_video(cfg.hw_width, cfg.hw_height,
                                                              cfg.sim_num_samples, cfg.rate_hz);
        if (cfg.kind == "imu")   return make_synthetic_imu(cfg.sim_num_samples, cfg.rate_hz);
        throw std::invalid_argument("Unknown sensor kind: " + cfg.kind);
    }

    if (cfg.mode == "hardware") {
        if (cfg.kind == "gps") {
            auto dev = cfg.hw_device.empty() ? "/dev/ttyUSB0" : cfg.hw_device;
            return std::make_unique<SerialGpsSensor>(dev, cfg.hw_baud);
        }
        if (cfg.kind == "video") {
            auto dev = cfg.hw_device.empty() ? "/dev/video0" : cfg.hw_device;
            return std::make_unique<V4L2CameraSensor>(dev, cfg.hw_width, cfg.hw_height, cfg.hw_fps);
        }
        if (cfg.kind == "imu") {
            auto dev = cfg.hw_device.empty() ? "/dev/spidev0.0" : cfg.hw_device;
            return std::make_unique<SpiImuSensor>(dev);
        }
        throw std::invalid_argument("Unknown sensor kind: " + cfg.kind);
    }

    throw std::invalid_argument("Unknown sensor mode: " + cfg.mode
                                + "  (expected 'simulated' or 'hardware')");
}

} // namespace hal
