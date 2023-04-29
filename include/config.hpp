#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include <string>

namespace config {
    const std::string ENGINE_PATH = "/home/zzy/meter_infer/engines/";
    const std::string IMAGE_PATH = "/home/zzy/meter_infer/data/images/";
    const std::string VIDEO_PATH = "/home/zzy/meter_infer/data/videos/";
    const std::string LOG_PATH = "/home/zzy/meter_infer/logs/";

    const int METER = 0; // pressure meter
    const int WATER = 1; // water level gauge
    const int LEVEL = 2; // water level
}

#endif