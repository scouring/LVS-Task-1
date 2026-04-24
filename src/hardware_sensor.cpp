#include "hardware_sensor.hpp"

#include <chrono>
#include <iostream>
#include <sstream>

// On a real target you would include platform headers here, e.g.:
//   #include <fcntl.h>
//   #include <termios.h>
//   #include <unistd.h>
//   #include <sys/ioctl.h>
//   #include <linux/videodev2.h>
//   #include <linux/spi/spidev.h>

namespace hal {

// ─────────────────────────────────────────────────────────────────────────────
//  HardwareSensor — base class
// ─────────────────────────────────────────────────────────────────────────────
HardwareSensor::HardwareSensor(std::string device_path,
                               std::string sensor_id,
                               float       rate_hz)
    : device_path_(std::move(device_path))
    , sensor_id_(std::move(sensor_id))
    , rate_hz_(rate_hz)
{}

HardwareSensor::~HardwareSensor() {
    if (ready_) close();
}

bool HardwareSensor::open() {
    std::cout << "[HardwareSensor] Opening " << device_path_ << " ...\n";

    // TODO: replace stub with real POSIX open():
    //   fd_ = ::open(device_path_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    //   if (fd_ < 0) { perror("open"); return false; }
    fd_ = -1;   // stub: no real fd

    if (!configure_device()) {
        std::cerr << "[HardwareSensor] configure_device() failed.\n";
        // TODO: ::close(fd_);
        fd_ = -1;
        return false;
    }

    ready_ = true;
    std::cout << "[HardwareSensor] " << description() << " ready (STUB).\n";
    return true;
}

void HardwareSensor::close() {
    if (fd_ >= 0) {
        // TODO: ::close(fd_);
        fd_ = -1;
    }
    ready_ = false;
    std::cout << "[HardwareSensor] " << sensor_id_ << " closed.\n";
}

std::optional<SensorSample> HardwareSensor::next_sample() {
    if (!ready_) return std::nullopt;

    auto raw     = read_raw();
    auto payload = decode(raw);
    if (!payload) return std::nullopt;

    SensorSample sample;
    sample.metadata.timestamp_us = now_us();
    sample.metadata.sequence_id  = sequence_id_++;
    sample.metadata.sensor_id    = sensor_id_;
    sample.metadata.is_simulated = false;
    sample.payload               = std::move(*payload);
    return sample;
}

bool HardwareSensor::is_ready() const { return ready_; }

std::string HardwareSensor::description() const {
    std::ostringstream oss;
    oss << "[HardwareSensor id=" << sensor_id_
        << " dev=" << device_path_
        << " rate=" << rate_hz_ << "Hz]";
    return oss.str();
}

std::vector<uint8_t> HardwareSensor::read_raw() {
    // TODO: platform read, e.g.:
    //   std::vector<uint8_t> buf(4096);
    //   ssize_t n = ::read(fd_, buf.data(), buf.size());
    //   if (n <= 0) return {};
    //   buf.resize(n);
    //   return buf;
    return {};  // stub
}

std::optional<SensorPayload> HardwareSensor::decode(
    const std::vector<uint8_t>& /*raw*/) {
    // TODO: protocol-specific decoding
    return std::nullopt;  // stub
}

uint64_t HardwareSensor::now_us() const {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<microseconds>(
            system_clock::now().time_since_epoch()).count());
}

// ─────────────────────────────────────────────────────────────────────────────
//  SerialGpsSensor
// ─────────────────────────────────────────────────────────────────────────────
SerialGpsSensor::SerialGpsSensor(const std::string& port, int baud)
    : HardwareSensor(port, "serial-gps", 10.f)
    , baud_(baud)
{}

bool SerialGpsSensor::configure_device() {
    std::cout << "[SerialGpsSensor] Configuring " << device_path_
              << " at " << baud_ << " baud (STUB).\n";
    // TODO:
    //   struct termios tty{};
    //   tcgetattr(fd_, &tty);
    //   cfsetospeed(&tty, B9600);   // map baud_ → Bxxx constant
    //   cfsetispeed(&tty, B9600);
    //   tty.c_cflag |= (CLOCAL | CREAD);
    //   tty.c_cflag &= ~PARENB;
    //   tty.c_cflag &= ~CSTOPB;
    //   tty.c_cflag &= ~CSIZE;
    //   tty.c_cflag |= CS8;
    //   tcsetattr(fd_, TCSANOW, &tty);
    return true;
}

std::vector<uint8_t> SerialGpsSensor::read_raw() {
    // TODO: read until '\n', accumulate into line_buffer_
    //   char c;
    //   while (::read(fd_, &c, 1) == 1) {
    //       if (c == '\n') break;
    //       line_buffer_ += c;
    //   }
    //   auto out = std::vector<uint8_t>(line_buffer_.begin(), line_buffer_.end());
    //   line_buffer_.clear();
    //   return out;
    return {};
}

std::optional<SensorPayload> SerialGpsSensor::decode(
    const std::vector<uint8_t>& raw) {
    if (raw.empty()) return std::nullopt;
    // TODO: parse NMEA sentence, e.g. $GPGGA or $GPRMC
    //   std::string sentence(raw.begin(), raw.end());
    //   if (sentence.rfind("$GPRMC", 0) != 0) return std::nullopt;
    //   … tokenise by ',' …
    //   GpsSample s{ … };
    //   return s;
    return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
//  V4L2CameraSensor
// ─────────────────────────────────────────────────────────────────────────────
V4L2CameraSensor::V4L2CameraSensor(const std::string& dev,
                                   uint32_t width,
                                   uint32_t height,
                                   float    fps)
    : HardwareSensor(dev, "v4l2-camera", fps)
    , width_(width), height_(height)
{}

bool V4L2CameraSensor::configure_device() {
    std::cout << "[V4L2CameraSensor] Configuring " << device_path_
              << " " << width_ << "x" << height_
              << " @ " << rate_hz_ << " fps (STUB).\n";
    // TODO:
    //   struct v4l2_format fmt{};
    //   fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    //   fmt.fmt.pix.width       = width_;
    //   fmt.fmt.pix.height      = height_;
    //   fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    //   fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
    //   ioctl(fd_, VIDIOC_S_FMT, &fmt);
    //   // Set up mmap buffers, start streaming …
    return true;
}

std::vector<uint8_t> V4L2CameraSensor::read_raw() {
    // TODO: dequeue mmap buffer, copy pixels, re-enqueue
    //   struct v4l2_buffer buf{};
    //   buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    //   buf.memory = V4L2_MEMORY_MMAP;
    //   ioctl(fd_, VIDIOC_DQBUF, &buf);
    //   auto data = std::vector<uint8_t>( mmap_ptrs_[buf.index],
    //                                     mmap_ptrs_[buf.index] + buf.bytesused );
    //   ioctl(fd_, VIDIOC_QBUF, &buf);
    //   return data;
    return {};
}

std::optional<SensorPayload> V4L2CameraSensor::decode(
    const std::vector<uint8_t>& raw) {
    if (raw.empty()) return std::nullopt;
    VideoFrame f;
    f.width    = width_;
    f.height   = height_;
    f.channels = 3;
    f.pixels   = raw;   // already RGB24 from V4L2
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  SpiImuSensor
// ─────────────────────────────────────────────────────────────────────────────
SpiImuSensor::SpiImuSensor(const std::string& dev, uint32_t speed_hz)
    : HardwareSensor(dev, "spi-imu", 100.f)
    , speed_hz_(speed_hz)
{}

bool SpiImuSensor::configure_device() {
    std::cout << "[SpiImuSensor] Configuring " << device_path_
              << " @ " << speed_hz_ << " Hz SPI (STUB).\n";
    // TODO:
    //   uint8_t  mode  = SPI_MODE_0;
    //   uint8_t  bits  = 8;
    //   uint32_t speed = speed_hz_;
    //   ioctl(fd_, SPI_IOC_WR_MODE,          &mode);
    //   ioctl(fd_, SPI_IOC_WR_BITS_PER_WORD, &bits);
    //   ioctl(fd_, SPI_IOC_WR_MAX_SPEED_HZ,  &speed);
    //   // Write ICM-42688 WHO_AM_I register, verify 0x47
    //   // Enable gyro + accel, set ODR, full-scale range
    return true;
}

std::vector<uint8_t> SpiImuSensor::read_raw() {
    // TODO: burst-read 14 bytes starting at ACCEL_DATA_X1 (0x1F)
    //   uint8_t tx[15] = { 0x1F | 0x80 };   // read bit set
    //   uint8_t rx[15] = {};
    //   struct spi_ioc_transfer xfer{ .tx_buf=(uint64_t)tx, .rx_buf=(uint64_t)rx,
    //                                 .len=15, .speed_hz=speed_hz_ };
    //   ioctl(fd_, SPI_IOC_MESSAGE(1), &xfer);
    //   return std::vector<uint8_t>(rx+1, rx+15);
    return {};
}

std::optional<SensorPayload> SpiImuSensor::decode(
    const std::vector<uint8_t>& raw) {
    if (raw.size() < 14) return std::nullopt;
    // TODO: combine high/low bytes per ICM-42688 register map,
    //       apply sensitivity scale factors.
    //   auto to_int16 = [&](int hi, int lo) {
    //       return static_cast<int16_t>((raw[hi]<<8) | raw[lo]);
    //   };
    //   const float ACCEL_SCALE = 9.81f / 2048.f;   // ±16g
    //   const float GYRO_SCALE  = 1.f / 131.f;       // ±250 dps → rad/s
    //   ImuSample s;
    //   s.accel_x = to_int16(0,1) * ACCEL_SCALE; ...
    return std::nullopt;
}

} // namespace hal
