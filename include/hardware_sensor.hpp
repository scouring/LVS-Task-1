#pragma once

#include "sensor_interface.hpp"
#include <string>

namespace hal {

// ─────────────────────────────────────────────────────────────────────────────
//  HardwareSensor  (skeleton / stub)
//
//  Drop-in replacement for SimulatedSensor once real hardware is available.
//  All platform-specific calls are clearly marked with TODO comments so an
//  embedded developer knows exactly what to fill in.
//
//  Typical concrete subclasses:
//    • SerialGpsSensor   — reads NMEA sentences over UART
//    • V4L2CameraSensor  — grabs frames from a Video4Linux2 device
//    • SpiImuSensor      — reads ICM-42688 over SPI via ioctl
// ─────────────────────────────────────────────────────────────────────────────
class HardwareSensor : public ISensorInput {
public:
    /// @param device_path  OS-level path to the device
    ///                     e.g. "/dev/ttyUSB0", "/dev/video0", "/dev/spidev0.0"
    /// @param sensor_id    Logical name used in metadata.
    /// @param rate_hz      Expected output rate; used by the pipeline scheduler.
    explicit HardwareSensor(std::string device_path,
                            std::string sensor_id = "hardware",
                            float       rate_hz   = 10.f);

    ~HardwareSensor() override;

    // ── ISensorInput ─────────────────────────────────────────────────────────
    bool open()   override;
    void close()  override;
    std::optional<SensorSample> next_sample() override;
    bool        is_ready()        const override;
    std::string description()     const override;
    float       nominal_rate_hz() const override { return rate_hz_; }

protected:
    // ── Extension points for subclasses ──────────────────────────────────────

    /// Called once by open() after the file-descriptor is obtained.
    /// Configure baud-rate, pixel format, SPI mode, etc. here.
    /// Return false to abort the open sequence.
    virtual bool configure_device() { return true; /* TODO: platform config */ }

    /// Read one raw buffer from the device (blocking or non-blocking).
    /// Returns empty vector on error / no data available.
    virtual std::vector<uint8_t> read_raw();

    /// Convert the raw bytes into a typed SensorPayload.
    /// Subclasses override this to implement their decoding logic.
    virtual std::optional<SensorPayload> decode(const std::vector<uint8_t>& raw);

    // ── Protected state available to subclasses ───────────────────────────────
    std::string device_path_;
    std::string sensor_id_;
    float       rate_hz_;
    int         fd_{-1};           ///< Generic file descriptor (POSIX)
    bool        ready_{false};
    uint32_t    sequence_id_{0};

private:
    uint64_t now_us() const;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Concrete hardware stubs — illustrate how a real driver is structured
// ─────────────────────────────────────────────────────────────────────────────

/// NMEA GPS over a serial port (e.g. u-blox M8).
class SerialGpsSensor final : public HardwareSensor {
public:
    explicit SerialGpsSensor(const std::string& port  = "/dev/ttyUSB0",
                             int                baud  = 9600);
protected:
    bool configure_device()                                  override;
    std::vector<uint8_t> read_raw()                         override;
    std::optional<SensorPayload> decode(const std::vector<uint8_t>&) override;
private:
    int baud_;
    std::string line_buffer_;
};

/// V4L2 camera (e.g. USB webcam or CSI ribbon).
class V4L2CameraSensor final : public HardwareSensor {
public:
    explicit V4L2CameraSensor(const std::string& dev    = "/dev/video0",
                              uint32_t           width  = 640,
                              uint32_t           height = 480,
                              float              fps    = 30.f);
protected:
    bool configure_device()                                  override;
    std::vector<uint8_t> read_raw()                         override;
    std::optional<SensorPayload> decode(const std::vector<uint8_t>&) override;
private:
    uint32_t width_, height_;
};

/// SPI IMU (e.g. ICM-42688-P).
class SpiImuSensor final : public HardwareSensor {
public:
    explicit SpiImuSensor(const std::string& dev      = "/dev/spidev0.0",
                          uint32_t           speed_hz = 1000000);
protected:
    bool configure_device()                                  override;
    std::vector<uint8_t> read_raw()                         override;
    std::optional<SensorPayload> decode(const std::vector<uint8_t>&) override;
private:
    uint32_t speed_hz_;
};

} // namespace hal
