#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>

using namespace std;
using namespace chrono;

int main()
{
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    PriorityQueue q;

    auto start_train = system_clock::now();

    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();

    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    unordered_set<string> test_set;
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
    int history = 0;
    const int generate_n = 10000000;

    auto start = system_clock::now();

    while (!q.Empty())
    {
        q.PopNext();

        q.total_guesses = (int)q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

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

        // 为了避免内存超限，达到一定数量后进行 Hash 和测试集匹配，然后清空 guesses。
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();

            bit32 state[4];

            for (const string &pw : q.guesses)
            {
                if (test_set.find(pw) != test_set.end())
                {
                    cracked += 1;
                }

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