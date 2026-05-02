#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe


static string stateToString(bit32 state[4])
{
    stringstream ss;
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        ss << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    return ss.str();
}


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    cout << "Testing MD5Hash correctness..." << endl;

    string test_pws[8] = {
        "123456",
        "password",
        "12345678",
        "qwerty",
        "123456789",
        "12345",
        "1234",
        "111111"
    };

    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };

    for (int i = 0; i < 8; i++)
    {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        string result = stateToString(state);

        if (result != test_hashes[i])
        {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << endl;
            cout << "Got:      " << result << endl;
            return 1;
        }
    }

    cout << "MD5Hash serial test passed!" << endl;

    cout << "Testing MD5Hash_SIMD4 correctness..." << endl;

    for (int base = 0; base < 8; base += 4)
    {
        string inputs[4];
        bit32 simd_states[4][4];
        bit32 serial_states[4][4];

        for (int i = 0; i < 4; i++)
        {
            inputs[i] = test_pws[base + i];
        }

        MD5Hash_SIMD4(inputs, simd_states);

        for (int i = 0; i < 4; i++)
        {
            MD5Hash(inputs[i], serial_states[i]);

            string simd_result = stateToString(simd_states[i]);
            string serial_result = stateToString(serial_states[i]);

            if (simd_result != serial_result)
            {
                cout << "MD5Hash_SIMD4 test failed for " << inputs[i] << "!" << endl;
                cout << "Serial: " << serial_result << endl;
                cout << "SIMD:   " << simd_result << endl;
                return 1;
            }
        }
    }

    string mixed_inputs[4] = {
        "abc",
        "nankai2026",
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva",
        "hello"
    };

    bit32 mixed_simd_states[4][4];
    bit32 mixed_serial_states[4][4];

    MD5Hash_SIMD4(mixed_inputs, mixed_simd_states);

    for (int i = 0; i < 4; i++)
    {
        MD5Hash(mixed_inputs[i], mixed_serial_states[i]);

        string simd_result = stateToString(mixed_simd_states[i]);
        string serial_result = stateToString(mixed_serial_states[i]);

        if (simd_result != serial_result)
        {
            cout << "MD5Hash_SIMD4 mixed-length test failed for input " << i << "!" << endl;
            cout << "Serial: " << serial_result << endl;
            cout << "SIMD:   " << simd_result << endl;
            return 1;
        }
    }

    cout << "MD5Hash_SIMD4 test passed!" << endl;

    bit32 state[4];
    MD5Hash("bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;

    return 0;
}