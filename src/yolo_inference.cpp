/**
 * YOLO Video Inference — C++ with OpenCV DNN
 *
 * Supports: YOLOv5, YOLOv8, YOLOv10 (ONNX format)
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && make -j$(nproc)
 *
 * Run:
 *   ./yolo_inference --model ../models/yolov8n.onnx \
 *                   --video path/to/video.mp4 \
 *                   --classes ../models/coco.names \
 *                   --conf 0.5 --nms 0.45
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

// ─── Configuration ────────────────────────────────────────────────────────────

struct Config {
    std::string modelPath;
    std::string videoPath;
    std::string classesPath;
    float       confThreshold = 0.50f;
    float       nmsThreshold  = 0.45f;
    int         inputWidth    = 640;
    int         inputHeight   = 640;
    bool        saveOutput    = false;
    std::string outputPath    = "output.mp4";
    bool        showWindow    = true;
    bool        useCuda       = false;
};

// ─── Detection Result ─────────────────────────────────────────────────────────

struct Detection {
    int   classId;
    float confidence;
    cv::Rect box;
};

// ─── YOLO Detector ────────────────────────────────────────────────────────────

class YOLODetector {
public:
    YOLODetector(const Config& cfg) : cfg_(cfg) {
        loadClasses();
        loadModel();
        initColors();
    }

    // Run detection on a single frame; returns annotated frame
    cv::Mat detect(const cv::Mat& frame) {
        cv::Mat blob = preprocess(frame);
        net_.setInput(blob);

        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        std::vector<Detection> dets = postprocess(outputs, frame.size());
        return drawDetections(frame.clone(), dets);
    }

    double getInferenceMs() const { return lastInferenceMs_; }

private:
    // ── Initialisation ────────────────────────────────────────────────────────

    void loadClasses() {
        if (cfg_.classesPath.empty()) {
            // Fallback: COCO class names inline
            classes_ = {
                "person","bicycle","car","motorcycle","airplane","bus","train",
                "truck","boat","traffic light","fire hydrant","stop sign",
                "parking meter","bench","bird","cat","dog","horse","sheep",
                "cow","elephant","bear","zebra","giraffe","backpack","umbrella",
                "handbag","tie","suitcase","frisbee","skis","snowboard",
                "sports ball","kite","baseball bat","baseball glove","skateboard",
                "surfboard","tennis racket","bottle","wine glass","cup","fork",
                "knife","spoon","bowl","banana","apple","sandwich","orange",
                "broccoli","carrot","hot dog","pizza","donut","cake","chair",
                "couch","potted plant","bed","dining table","toilet","tv",
                "laptop","mouse","remote","keyboard","cell phone","microwave",
                "oven","toaster","sink","refrigerator","book","clock","vase",
                "scissors","teddy bear","hair drier","toothbrush"
            };
            return;
        }
        std::ifstream file(cfg_.classesPath);
        if (!file.is_open())
            throw std::runtime_error("Cannot open classes file: " + cfg_.classesPath);
        std::string line;
        while (std::getline(file, line))
            if (!line.empty()) classes_.push_back(line);
    }

    void loadModel() {
        net_ = cv::dnn::readNetFromONNX(cfg_.modelPath);
        if (net_.empty())
            throw std::runtime_error("Failed to load model: " + cfg_.modelPath);

        if (cfg_.useCuda) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "[INFO] Using CUDA backend\n";
        } else {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "[INFO] Using CPU backend\n";
        }
        std::cout << "[INFO] Model loaded: " << cfg_.modelPath << "\n";
        std::cout << "[INFO] Classes: " << classes_.size() << "\n";
    }

    void initColors() {
        cv::RNG rng(42);
        colors_.resize(classes_.size());
        for (auto& c : colors_)
            c = cv::Scalar(rng.uniform(50,255), rng.uniform(50,255), rng.uniform(50,255));
    }

    // ── Pre / Post Processing ─────────────────────────────────────────────────

    cv::Mat preprocess(const cv::Mat& frame) {
        origSize_ = frame.size();
        // Letterbox resize to preserve aspect ratio
        float scale = std::min(
            static_cast<float>(cfg_.inputWidth)  / frame.cols,
            static_cast<float>(cfg_.inputHeight) / frame.rows);
        int newW = static_cast<int>(frame.cols * scale);
        int newH = static_cast<int>(frame.rows * scale);
        padX_ = (cfg_.inputWidth  - newW) / 2;
        padY_ = (cfg_.inputHeight - newH) / 2;
        scale_ = scale;

        cv::Mat resized;
        cv::resize(frame, resized, {newW, newH});

        cv::Mat padded(cfg_.inputHeight, cfg_.inputWidth, CV_8UC3, cv::Scalar(114,114,114));
        resized.copyTo(padded(cv::Rect(padX_, padY_, newW, newH)));

        return cv::dnn::blobFromImage(
            padded, 1.0/255.0,
            {cfg_.inputWidth, cfg_.inputHeight},
            cv::Scalar(0,0,0), true, false, CV_32F);
    }

    // Supports both YOLOv5 and YOLOv8 output layouts
    std::vector<Detection> postprocess(const std::vector<cv::Mat>& outputs,
                                       cv::Size /*frameSize*/) {
        auto t0 = std::chrono::steady_clock::now();

        std::vector<int>    classIds;
        std::vector<float>  confidences;
        std::vector<cv::Rect> boxes;

        const cv::Mat& out = outputs[0];

        // YOLOv8: shape [1, 84, 8400] → transpose to [8400, 84]
        // YOLOv5: shape [1, 25200, 85]
        bool isYOLOv8 = (out.dims == 3 && out.size[2] > out.size[1]);

        cv::Mat data = out.reshape(1, out.total() / out.size[out.dims - 1]);
        if (isYOLOv8) cv::transpose(data, data); // [num_det, 4+classes]

        int numClasses = static_cast<int>(classes_.size());

        for (int i = 0; i < data.rows; ++i) {
            const float* row = data.ptr<float>(i);

            float cx = row[0], cy = row[1], w = row[2], h = row[3];

            // YOLOv5 has objectness at row[4]; YOLOv8 does not
            int scoreOffset = isYOLOv8 ? 4 : 5;
            float objConf   = isYOLOv8 ? 1.0f : row[4];

            // Find best class
            int   bestClass = -1;
            float bestScore = 0.f;
            for (int c = 0; c < numClasses; ++c) {
                float s = row[scoreOffset + c] * objConf;
                if (s > bestScore) { bestScore = s; bestClass = c; }
            }
            if (bestScore < cfg_.confThreshold) continue;

            // Map from model space → original frame
            int x1 = static_cast<int>((cx - w/2 - padX_) / scale_);
            int y1 = static_cast<int>((cy - h/2 - padY_) / scale_);
            int bw = static_cast<int>(w / scale_);
            int bh = static_cast<int>(h / scale_);

            // Clamp
            x1 = std::clamp(x1, 0, origSize_.width);
            y1 = std::clamp(y1, 0, origSize_.height);
            bw = std::clamp(bw, 1, origSize_.width  - x1);
            bh = std::clamp(bh, 1, origSize_.height - y1);

            classIds.push_back(bestClass);
            confidences.push_back(bestScore);
            boxes.emplace_back(x1, y1, bw, bh);
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, cfg_.confThreshold, cfg_.nmsThreshold, indices);

        std::vector<Detection> dets;
        dets.reserve(indices.size());
        for (int idx : indices)
            dets.push_back({classIds[idx], confidences[idx], boxes[idx]});

        auto t1 = std::chrono::steady_clock::now();
        lastInferenceMs_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

        return dets;
    }

    // ── Drawing ───────────────────────────────────────────────────────────────

    cv::Mat drawDetections(cv::Mat frame, const std::vector<Detection>& dets) {
        for (const auto& d : dets) {
            const cv::Scalar& color = colors_[d.classId % colors_.size()];
            cv::rectangle(frame, d.box, color, 2);

            std::string label = classes_[d.classId] + " " +
                                cv::format("%.2f", d.confidence);

            int baseLine = 0;
            cv::Size ts  = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseLine);
            int tx = d.box.x;
            int ty = std::max(d.box.y - 5, ts.height + 4);

            cv::rectangle(frame,
                {tx, ty - ts.height - 4},
                {tx + ts.width + 2, ty + baseLine},
                color, cv::FILLED);
            cv::putText(frame, label, {tx + 1, ty - 1},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1);
        }
        return frame;
    }

    // ── Members ───────────────────────────────────────────────────────────────

    Config                   cfg_;
    cv::dnn::Net             net_;
    std::vector<std::string> classes_;
    std::vector<cv::Scalar>  colors_;
    cv::Size                 origSize_;
    float                    scale_ = 1.f;
    int                      padX_  = 0;
    int                      padY_  = 0;
    double                   lastInferenceMs_ = 0.0;
};

// ─── CLI Parsing ──────────────────────────────────────────────────────────────

void printUsage(const char* prog) {
    std::cout
        << "\nUsage: " << prog << " [OPTIONS]\n\n"
        << "Required:\n"
        << "  --model   <path>   ONNX model file (e.g. yolov8n.onnx)\n"
        << "  --video   <path>   Input video file (or '0' for webcam)\n\n"
        << "Optional:\n"
        << "  --classes <path>   Newline-delimited class names (default: COCO built-in)\n"
        << "  --conf    <float>  Confidence threshold (default: 0.5)\n"
        << "  --nms     <float>  NMS IoU threshold (default: 0.45)\n"
        << "  --width   <int>    Model input width  (default: 640)\n"
        << "  --height  <int>    Model input height (default: 640)\n"
        << "  --save    <path>   Save annotated video to this path\n"
        << "  --noshow           Do not open display window\n"
        << "  --cuda             Use CUDA backend (requires CUDA-enabled OpenCV)\n\n";
}

Config parseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + a);
            return argv[++i];
        };
        if      (a == "--model")   cfg.modelPath    = next();
        else if (a == "--video")   cfg.videoPath    = next();
        else if (a == "--classes") cfg.classesPath  = next();
        else if (a == "--conf")    cfg.confThreshold = std::stof(next());
        else if (a == "--nms")     cfg.nmsThreshold  = std::stof(next());
        else if (a == "--width")   cfg.inputWidth    = std::stoi(next());
        else if (a == "--height")  cfg.inputHeight   = std::stoi(next());
        else if (a == "--save")  { cfg.saveOutput = true; cfg.outputPath = next(); }
        else if (a == "--noshow")  cfg.showWindow  = false;
        else if (a == "--cuda")    cfg.useCuda     = true;
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); std::exit(0); }
        else std::cerr << "[WARN] Unknown argument: " << a << "\n";
    }
    if (cfg.modelPath.empty() || cfg.videoPath.empty()) {
        printUsage(argv[0]);
        throw std::invalid_argument("--model and --video are required");
    }
    return cfg;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Config cfg;
    try {
        cfg = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    // Open video
    cv::VideoCapture cap;
    if (cfg.videoPath == "0" || cfg.videoPath == "webcam")
        cap.open(0);
    else
        cap.open(cfg.videoPath);

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video: " << cfg.videoPath << "\n";
        return 1;
    }

    int    frameW  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int    frameH  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps     = cap.get(cv::CAP_PROP_FPS);
    int    total   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "[INFO] Video: " << frameW << "x" << frameH
              << " @ " << fps << " fps, " << total << " frames\n";

    // Optional output writer
    cv::VideoWriter writer;
    if (cfg.saveOutput) {
        writer.open(cfg.outputPath,
                    cv::VideoWriter::fourcc('m','p','4','v'),
                    fps, {frameW, frameH});
        if (!writer.isOpened())
            std::cerr << "[WARN] Cannot open output video: " << cfg.outputPath << "\n";
        else
            std::cout << "[INFO] Writing output to: " << cfg.outputPath << "\n";
    }

    // Build detector
    YOLODetector detector(cfg);

    // ── Main loop ──────────────────────────────────────────────────────────────
    cv::Mat frame;
    int frameIdx   = 0;
    double totalMs = 0.0;
    auto wallStart = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
        ++frameIdx;

        cv::Mat annotated = detector.detect(frame);
        double  ms        = detector.getInferenceMs();
        totalMs += ms;

        // FPS overlay
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wallStart).count();
        double realFps = frameIdx / elapsed;

        cv::putText(annotated,
                    cv::format("FPS: %.1f  Inf: %.1fms", realFps, ms),
                    {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2);

        if (cfg.showWindow) {
            cv::imshow("YOLO Inference", annotated);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;   // q or ESC to quit
            if (key == ' ') cv::waitKey(0);        // space to pause
        }

        if (writer.isOpened()) writer.write(annotated);

        if (frameIdx % 30 == 0)
            std::cout << "\r[INFO] Frame " << frameIdx << "/" << total
                      << "  avg inf: " << cv::format("%.1f", totalMs/frameIdx)
                      << "ms  FPS: " << cv::format("%.1f", realFps) << std::flush;
    }

    std::cout << "\n[INFO] Done. Processed " << frameIdx << " frames. "
              << "Avg inference: " << cv::format("%.2f", totalMs/frameIdx) << "ms\n";
    return 0;
}
