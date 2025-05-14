#include <fstream>
#include "json.hpp"
#include "detect/hardware_detect.hpp"


using json = nlohmann::json;

HardwareType detectHardwareFromConfig() {
    std::ifstream file("config.json");
    if (!file.is_open()) return HardwareType::CPU;
    json j;
    file >> j;
    std::string hw = j.value("kernel", "cpu");
    if (hw == "gpu") return HardwareType::GPU;
    if (hw == "simd") return HardwareType::SIMD;
    return HardwareType::CPU;
}

int getThreadCountFromConfig() {
    std::ifstream file("config.json");
    if (!file.is_open()) return 1;
    json j;
    file >> j;
    return j.value("threads", 1);
}
