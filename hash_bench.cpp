#include "md5.h"
#include <iostream>
#include <string>
#include <chrono>
#include <array>
using namespace std;
using namespace chrono;

static unsigned long long state_checksum(bit32 state[4]) {
    return (unsigned long long)state[0] + state[1] + state[2] + state[3];
}

int main() {
    // 都选短口令，填充后通常都是 1 个 512-bit block，避免长度不一致回退
    array<string, 16> inputs = {
        "password", "123456", "nankai2026", "hello",
        "admin123", "qwerty", "testtest", "abc123",
        "student", "computer", "parallel", "simd2026",
        "md5test", "openEuler", "kunpeng", "hashonly"
    };

    const int repeat = 200000; // 每轮 16 个口令，总共 3200000 次哈希
    volatile unsigned long long checksum_serial = 0;
    volatile unsigned long long checksum_simd = 0;

    bit32 state[4];

    auto start_serial = high_resolution_clock::now();

    for (int r = 0; r < repeat; r++) {
        for (int i = 0; i < 16; i++) {
            MD5Hash(inputs[i], state);
            checksum_serial += state_checksum(state);
        }
    }

    auto end_serial = high_resolution_clock::now();
    double time_serial = duration<double>(end_serial - start_serial).count();

    auto start_simd = high_resolution_clock::now();

    for (int r = 0; r < repeat; r++) {
        for (int i = 0; i < 16; i += 4) {
            string batch[4] = {
                inputs[i],
                inputs[i + 1],
                inputs[i + 2],
                inputs[i + 3]
            };

            bit32 states[4][4];
            MD5Hash_SIMD4(batch, states);

            for (int j = 0; j < 4; j++) {
                checksum_simd += state_checksum(states[j]);
            }
        }
    }

    auto end_simd = high_resolution_clock::now();
    double time_simd = duration<double>(end_simd - start_simd).count();

    cout << "Hash-only serial time: " << time_serial << " seconds" << endl;
    cout << "Hash-only SIMD4 time: " << time_simd << " seconds" << endl;
    cout << "Hash-only speedup: " << time_serial / time_simd << endl;
    cout << "checksum serial: " << checksum_serial << endl;
    cout << "checksum SIMD4: " << checksum_simd << endl;

    return 0;
}
