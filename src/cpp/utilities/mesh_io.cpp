#include "mesh_io.hpp"
#include <string>
#include <ctime>

#ifndef __TIDY_CUR
#define __TIDY_CUR
std::string __tidy_cur_time() {
    static char date[24];
    auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::strftime(date, sizeof(date), "%H:%M:%S", std::gmtime(&in_time_t));
    return date;
}
#endif