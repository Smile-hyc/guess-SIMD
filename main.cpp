#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <cstdlib>
using namespace std;
using namespace chrono;

// 编译指令：
// g++ -O2 -fopenmp main.cpp train.cpp guessing.cpp md5.cpp -o main_openmp

int main(int argc, char* argv[])
{
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    PriorityQueue q;

    if (argc >= 2)
    {
        q.omp_thread_num = atoi(argv[1]);

        if (q.omp_thread_num < 1)
        {
            q.omp_thread_num = 1;
        }
    }

    if (argc >= 3)
    {
        q.omp_threshold = atoi(argv[2]);

        if (q.omp_threshold < 0)
        {
            q.omp_threshold = 0;
        }
    }

    if (argc >= 4)
    {
        q.omp_chunk_size = atoi(argv[3]);

        if (q.omp_chunk_size <= 0)
        {
            q.omp_chunk_size = 1;
        }
    }

    cout << "OpenMP threads: " << q.omp_thread_num << endl;
    cout << "OpenMP threshold: " << q.omp_threshold << endl;
    cout << "OpenMP schedule: dynamic" << endl;
    cout << "OpenMP chunk size: " << q.omp_chunk_size << endl;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();

    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();

    cout << "here" << endl;

    int curr_num = 0;
    int history = 0;

    auto start = system_clock::now();

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            int generate_n = 10000000;

            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;

                break;
            }
        }

        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();

            bit32 state[4];

            for (string pw : q.guesses)
            {
                MD5Hash(pw, state);
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    return 0;
}