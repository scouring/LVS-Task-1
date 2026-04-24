// main.cpp — HAL demo: inference pipeline consuming ISensorInput
//
// Build (from hal/):
//   g++ -std=c++17 -Iinclude src/*.cpp main.cpp -o hal_demo
//
// Run examples:
//   ./hal_demo                          # synthetic GPS (default)
//   ./hal_demo mode=simulated kind=imu
//   ./hal_demo mode=simulated kind=video sim_num_samples=5
//   ./hal_demo mode=simulated kind=gps  sim_file=gps_log.csv
//   ./hal_demo mode=hardware  kind=gps  hw_device=/dev/ttyUSB0
//   ./hal_demo config=sensor.cfg        # load from file

#include "sensor_interface.hpp"
#include "sensor_factory.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>

// ─────────────────────────────────────────────────────────────────────────────
//  Pretty-printers for each payload type
// ─────────────────────────────────────────────────────────────────────────────
static void print_payload(const hal::GpsSample& g) {
    std::cout << std::fixed << std::setprecision(6)
              << "  GPS  lat=" << g.latitude_deg
              << " lon="       << g.longitude_deg
              << " alt="       << std::setprecision(1) << g.altitude_m << "m"
              << " spd="       << g.speed_mps     << "m/s"
              << " hdg="       << g.heading_deg   << "°\n";
}

static void print_payload(const hal::VideoFrame& f) {
    // Only show a 2x2 corner sample — printing all pixels would be unwieldy
    std::cout << "  IMG  " << f.width << "x" << f.height
              << " ch=" << f.channels
              << "  top-left px=["
              << static_cast<int>(f.pixels[0]) << ","
              << static_cast<int>(f.pixels[1]) << ","
              << static_cast<int>(f.pixels[2]) << "]\n";
}

static void print_payload(const hal::ImuSample& s) {
    std::cout << std::fixed << std::setprecision(4)
              << "  IMU  accel=["
              << s.accel_x << "," << s.accel_y << "," << s.accel_z << "] m/s²"
              << "  gyro=["
              << s.gyro_x  << "," << s.gyro_y  << "," << s.gyro_z  << "] rad/s\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Inference pipeline — knows NOTHING about simulated vs. hardware
// ─────────────────────────────────────────────────────────────────────────────
class InferencePipeline {
public:
    explicit InferencePipeline(hal::SensorPtr sensor, uint32_t max_samples = 0)
        : sensor_(std::move(sensor)), max_samples_(max_samples) {}

    void run() {
        if (!sensor_->open()) {
            std::cerr << "[Pipeline] Failed to open sensor.\n";
            return;
        }

        std::cout << "\n[Pipeline] Starting — source: "
                  << sensor_->description() << "\n"
                  << std::string(60, '-') << "\n";

        uint32_t count = 0;
        const float rate = sensor_->nominal_rate_hz();
        const int delay_ms = (rate > 0.f)
            ? static_cast<int>(1000.f / rate)
            : 0;

        while (sensor_->is_ready()) {
            if (max_samples_ > 0 && count >= max_samples_) break;

            auto maybe = sensor_->next_sample();
            if (!maybe) break;

            const auto& sample = *maybe;
            std::cout << "[" << std::setw(4) << sample.metadata.sequence_id << "]"
                      << " ts=" << sample.metadata.timestamp_us << "µs"
                      << " src=" << sample.metadata.sensor_id
                      << (sample.metadata.is_simulated ? " (sim)" : " (hw)") << "\n";

            // Dispatch to type-specific handler via std::visit
            std::visit([](const auto& p) { print_payload(p); }, sample.payload);

            // Simulate pipeline processing time / rate limiting
            if (delay_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

            ++count;
        }

        std::cout << std::string(60, '-') << "\n"
                  << "[Pipeline] Done — processed " << count << " sample(s).\n\n";
        sensor_->close();
    }

private:
    hal::SensorPtr sensor_;
    uint32_t       max_samples_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Parse CLI args:  key=value  or  config=<file>
// ─────────────────────────────────────────────────────────────────────────────
static std::unordered_map<std::string,std::string>
parse_args(int argc, char* argv[]) {
    std::unordered_map<std::string,std::string> m;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        if (eq != std::string::npos)
            m[arg.substr(0,eq)] = arg.substr(eq+1);
    }
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::cout << "========================================================\n"
              << "=          Sensor HAL — Pipeline Demo                  =\n"
              << "========================================================\n\n";

    auto args = parse_args(argc, argv);

    // ── Load configuration ────────────────────────────────────────────────────
    hal::SensorConfig cfg;
    if (args.count("config")) {
        try {
            cfg = hal::SensorConfig::from_file(args.at("config"));
        } catch (const std::exception& e) {
            std::cerr << "[main] " << e.what() << "\n";
            return EXIT_FAILURE;
        }
    } else {
        // Apply any key=value overrides from the command line
        cfg = hal::SensorConfig::from_map(args);
    }

    cfg.print();

    // ── Demo 1: Run with the configured source ────────────────────────────────
    {
        std::cout << "\n=== Demo 1: Configured source ===\n";
        try {
            auto sensor = hal::SensorFactory::create(cfg);
            InferencePipeline pipeline(std::move(sensor), /*max_samples=*/8);
            pipeline.run();
        } catch (const std::exception& e) {
            std::cerr << "[main] Factory error: " << e.what() << "\n";
            return EXIT_FAILURE;
        }
    }

    // ── Demo 2: Swap to a different synthetic source — zero pipeline changes ──
    if (cfg.mode == "simulated") {
        std::cout << "=== Demo 2: Swap to synthetic IMU (no pipeline change) ===\n";
        hal::SensorConfig imu_cfg;
        imu_cfg.kind        = "imu";
        imu_cfg.mode        = "simulated";
        imu_cfg.sim_num_samples = 6;
        imu_cfg.rate_hz     = 100.f;

        imu_cfg.print();
        auto sensor = hal::SensorFactory::create(imu_cfg);
        InferencePipeline pipeline(std::move(sensor));
        pipeline.run();
    }

    // ── Demo 3: Rewind a file-backed source and replay ───────────────────────
    if (cfg.mode == "simulated" && !cfg.sim_file.empty()) {
        std::cout << "=== Demo 3: Rewind file-backed source and replay ===\n";
        auto sensor = hal::SensorFactory::create(cfg);
        if (sensor->open()) {
            // First pass
            uint32_t n = 0;
            while (sensor->is_ready() && n < 3) { sensor->next_sample(); ++n; }
            std::cout << "[main] Read " << n << " samples, rewinding...\n";
            sensor->rewind();
            // Second pass from the beginning
            InferencePipeline pipeline(std::move(sensor), 3);
            pipeline.run();
        }
    }

    std::cout << "All demos complete.\n";
    return EXIT_SUCCESS;
}
