#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>

using namespace std;
using namespace chrono;

// 编译指令：
// mpic++ correctness_guess_mpi_pt.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    PriorityQueue q;
    q.mpi_rank = rank;
    q.mpi_size = size;

    auto start_train = system_clock::now();

    // 为了保证各进程拥有相同的模型和优先队列，基础版本中每个进程都训练一份模型。
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
    int total_cracked = 0;

    q.init();

    if (rank == 0)
    {
        cout << "Starting PT-level MPI version, mpi_size = " << size << endl;
    }

    const int BATCH_SIZE = size; // 一次最多取出 size 个 PT

    int curr_num = 0;
    int history = 0;
    bool should_continue = true;

    size_t last_local_guess_size = 0;

    auto start = system_clock::now();

    while (should_continue)
    {
        int local_has_work = q.priority.empty() ? 0 : 1;
        int global_has_work = 0;

        MPI_Allreduce(&local_has_work, &global_has_work,
                      1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_has_work == 0)
        {
            should_continue = false;
            break;
        }

        if (!q.priority.empty())
        {
            q.PopNextBatch(BATCH_SIZE);
        }

        // 只统计本轮新增的 guesses，避免反复累计整个 q.guesses.size()
        int local_added = (int)(q.guesses.size() - last_local_guess_size);
        last_local_guess_size = q.guesses.size();

        int global_added = 0;

        MPI_Allreduce(&local_added, &global_added,
                      1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        curr_num += global_added;

        int should_exit = 0;

        if (rank == 0 && curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + curr_num << endl;

            if (history + curr_num > 10000000)
            {
                should_exit = 1;
            }
        }

        MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (should_exit)
        {
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

            MPI_Reduce(&cracked, &total_cracked,
                       1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                cout << "Cracked:" << total_cracked << endl;
            }

            should_continue = false;
            break;
        }

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

                MD5Hash(pw, state);
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            history += curr_num;
            curr_num = 0;

            q.guesses.clear();
            last_local_guess_size = 0;
        }
    }

    MPI_Finalize();
    return 0;
}