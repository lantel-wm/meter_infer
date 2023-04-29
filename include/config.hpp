#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>

namespace config {
    const std::string IMAGE_PATH = "/home/zzy/meter_infer/data/images/";
    const std::string VIDEO_PATH = "/home/zzy/meter_infer/data/videos/";
    const std::string LOG_PATH = "/home/zzy/meter_infer/logs/";

    const int METER = 0; // pressure meter
    const int LEVEL = 1; // water level gauge
}

#endif