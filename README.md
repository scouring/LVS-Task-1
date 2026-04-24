# Sensor HAL — Hardware Abstraction Layer Demo

A C++17 library that lets an inference pipeline consume sensor data (GPS, camera, IMU) without caring whether the source is real hardware or a simulation. Swap sources at runtime with no pipeline code changes.

## Project structure

```
HAL/
├── include/
│   ├── sensor_interface.hpp   # Abstract ISensorInput contract
│   ├── sensor_factory.hpp     # SensorConfig + SensorFactory
│   ├── simulated_sensor.hpp   # Simulated source (synthetic + file-backed)
│   └── hardware_sensor.hpp    # Hardware stubs (serial GPS, V4L2, SPI IMU)
├── src/
│   ├── simulated_sensor.cpp
│   ├── hardware_sensor.cpp
│   └── sensor_factory.cpp
├── main.cpp                   # Demo: three pipeline runs back-to-back
├── sensor.cfg                 # Example config file
└── gps_log.csv                # Pre-recorded GPS track for file-based replay
```

## Build

Requires a C++17 compiler (GCC 8+ or Clang 7+). No external dependencies.

```bash
g++ -std=c++17 -Iinclude src/*.cpp main.cpp -o hal_demo
```

## Run

```bash
# Default: synthetic GPS, 100 samples at 10 Hz
./hal_demo

# Specific sensor kind
./hal_demo mode=simulated kind=imu
./hal_demo mode=simulated kind=video sim_num_samples=5

# File-backed GPS replay
./hal_demo mode=simulated kind=gps sim_file=gps_log.csv

# Hardware mode (requires a real device)
./hal_demo mode=hardware kind=gps hw_device=/dev/ttyUSB0

# Load all settings from a config file
./hal_demo config=sensor.cfg
```

## Configuration

All options can be supplied as `key=value` CLI arguments or in a `key = value` config file (lines starting with `#` are comments).

| Key | Default | Description |
|---|---|---|
| `mode` | `simulated` | `simulated` or `hardware` |
| `kind` | `gps` | `gps`, `video`, or `imu` |
| `rate_hz` | `10` | Nominal output rate (Hz) |
| `sim_file` | *(empty)* | CSV path for file-backed replay; blank = synthetic generator |
| `sim_num_samples` | `100` | Samples to generate in synthetic mode |
| `hw_device` | `/dev/ttyUSB0` | OS device path for hardware mode |
| `hw_baud` | `9600` | Baud rate (serial GPS) |
| `hw_width` / `hw_height` | `640` / `480` | Resolution (camera) |
| `hw_fps` | `30` | Frame rate (camera) |

## Sensor types

| Kind | Payload fields |
|---|---|
| `gps` | `latitude_deg`, `longitude_deg`, `altitude_m`, `speed_mps`, `heading_deg` |
| `video` | `width`, `height`, `channels`, `pixels` (row-major RGB) |
| `imu` | `accel_{x,y,z}` (m/s²), `gyro_{x,y,z}` (rad/s), `mag_{x,y,z}` (µT) |

Every sample also carries metadata: `timestamp_us`, `sequence_id`, `sensor_id`, `is_simulated`.

## Simulated sources

| Factory helper | Description |
|---|---|
| `make_synthetic_gps()` | Circular GPS track (~111 m radius) around a configurable centre point |
| `make_synthetic_video()` | Animated RGB checkerboard at configurable resolution |
| `make_synthetic_imu()` | Sinusoidal accel/gyro data simulating vibration |
| `make_file_gps(path)` | Replays a CSV file (`lat,lon,alt,speed,heading`); supports `rewind()` |

## Hardware stubs

`HardwareSensor` is a base class with TODO markers showing exactly what platform code to fill in. Three concrete stubs are provided:

| Class | Device | Protocol |
|---|---|---|
| `SerialGpsSensor` | `/dev/ttyUSB0` | NMEA sentences over UART |
| `V4L2CameraSensor` | `/dev/video0` | Video4Linux2 mmap streaming |
| `SpiImuSensor` | `/dev/spidev0.0` | ICM-42688 burst-read over SPI |

To implement a real driver, subclass `HardwareSensor` and override `configure_device()`, `read_raw()`, and `decode()`.

## Adding a new sensor type

1. Add a new payload struct to `sensor_interface.hpp` and include it in the `SensorPayload` variant.
2. Add a `print_payload()` overload in `main.cpp` (or wherever you process samples).
3. Add a factory helper in `simulated_sensor.cpp` for the synthetic source.
4. Subclass `HardwareSensor` for the real driver.
5. Wire the new `kind` string into `SensorFactory::create()`.

The pipeline (`InferencePipeline` in `main.cpp`) needs no changes — `std::visit` dispatches automatically to the new handler.