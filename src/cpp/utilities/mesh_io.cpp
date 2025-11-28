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

#ifndef __PROGBAR
#define __PROGBAR
#include <stdio.h>

static auto start = std::chrono::high_resolution_clock::now();
static auto true_begin = start;
static double secp = 0.0;

void
__update_prog(const int p, const int n, const int each, const char* c) {
    static const char* hash = "#########################";
    static const char* dash = "-------------------------";
    int prog_i;
    if (p == 0) {
        printf("%s", c); // Flush the color of the bar
        printf("[--------------------] 0%% ETA: n\\a   ");
        fflush(stdout);
        start = std::chrono::high_resolution_clock::now();
        true_begin = start;
    }
    else {
        printf("\r");
        auto finish = std::chrono::high_resolution_clock::now();
        long ms_val = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        prog_i = (20 * p) / n;
        start = finish;
        if (prog_i > 20)
            prog_i = 20;
        printf("[%.*s%.*s] %d%% ", prog_i, hash, 20 - prog_i, dash, (100*p) /n);
        double s = (ms_val * ((double)n - p) / each) / 1000.0;
        printf("ETA: %.3lf sec.   ", 0.8*s + 0.2*secp);
        secp = s;
        fflush(stdout);
    }

}

void
__terminate_prog() {
    const auto now = std::chrono::high_resolution_clock::now();
    long ms_val = std::chrono::duration_cast<std::chrono::milliseconds>(now - true_begin).count();
    printf("\r[####################] Task completed succesfully in [%lf sec.]", ms_val / 1000.0);
    printf("\n");
    secp = 0.0;
}

#endif
