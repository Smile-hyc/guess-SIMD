#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <cstdlib>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ -O2 -fopenmp correctness_guess_openmp.cpp train.cpp guessing.cpp md5.cpp -o correctness_guess_openmp

static int parse_schedule_type(const string& s)
{
    if (s == "static")
    {
        return 0;
    }
    if (s == "dynamic")
    {
        return 1;
    }
    if (s == "guided")
    {
        return 2;
    }
    return 0;
}

static string schedule_name(int type)
{
    if (type == 0)
    {
        return "static";
    }
    if (type == 1)
    {
        return "dynamic";
    }
    if (type == 2)
    {
        return "guided";
    }
    return "static";
}

int main(int argc, char* argv[])
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;

    // argv[1]：线程数，例如 ./correctness_guess_openmp 8
    // argv[2]：并行阈值，例如 ./correctness_guess_openmp 8 1000
    // argv[3]：调度策略 static / dynamic / guided
    // argv[4]：chunk size
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
        q.omp_schedule_type = parse_schedule_type(argv[3]);
    }

    if (argc >= 5)
    {
        q.omp_chunk_size = atoi(argv[4]);
        if (q.omp_chunk_size <= 0)
        {
            q.omp_chunk_size = 1;
        }
    }

    cout << "OpenMP threads: " << q.omp_thread_num << endl;
    cout << "OpenMP threshold: " << q.omp_threshold << endl;
    cout << "OpenMP schedule: " << schedule_name(q.omp_schedule_type) << endl;
    cout << "OpenMP chunk size: " << q.omp_chunk_size << endl;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // 加载测试集，用于统计 Cracked
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw;

    while (test_data >> pw)
    {
        test_count += 1;
        test_set.insert(pw);

        if (test_count >= 1000000)
        {
            break;
        }
    }

    int cracked = 0;

    q.init();

    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();

    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results.txt");

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                cout << "Cracked:" << cracked << endl;

                break;
            }
        }

        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            bit32 state[4];

            for (string pw : q.guesses)
            {
                if (test_set.find(pw) != test_set.end())
                {
                    cracked += 1;
                }

                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // a<<pw<<"\t";
                // for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                // }
                // a << endl;
            }

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    return 0;
}