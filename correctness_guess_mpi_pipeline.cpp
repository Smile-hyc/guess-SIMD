#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

using namespace std;
using namespace chrono;

// 生成--哈希流水线共享缓存
mutex guesses_mutex;
vector<string> pending_hash_guesses;

// 哈希线程退出标志
atomic<bool> hash_thread_should_exit(false);

// 本进程内已经破解的数量
atomic<int> local_cracked(0);

// 本进程实际完成哈希的数量，主要用于调试和后续分析
atomic<int> local_hashed(0);

// 将主线程生成的 guesses 转交给哈希线程
void push_guesses_to_hash_queue(vector<string> &guesses)
{
    if (guesses.empty())
    {
        return;
    }

    lock_guard<mutex> lock(guesses_mutex);

    pending_hash_guesses.insert(
        pending_hash_guesses.end(),
        guesses.begin(),
        guesses.end());

    guesses.clear();
}

// 哈希线程函数：不断从 pending_hash_guesses 中取出候选口令并计算 MD5
void hash_worker_thread(const unordered_set<string> &test_set, double &time_hash)
{
    while (true)
    {
        vector<string> local_guesses;

        {
            lock_guard<mutex> lock(guesses_mutex);

            if (!pending_hash_guesses.empty())
            {
                local_guesses.swap(pending_hash_guesses);
            }
            else if (hash_thread_should_exit)
            {
                break;
            }
        }

        if (local_guesses.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }

        auto start_hash = system_clock::now();

        bit32 state[4];

        for (const string &pw : local_guesses)
        {
            if (test_set.find(pw) != test_set.end())
            {
                local_cracked++;
            }

            MD5Hash(pw, state);
            local_hashed++;
        }

        auto end_hash = system_clock::now();
        auto duration_hash = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration_hash.count()) *
                     microseconds::period::num /
                     microseconds::period::den;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;   // 本进程哈希计算时间
    double time_guess = 0;  // 本进程生成计算时间
    double time_train = 0;  // 本进程训练时间

    PriorityQueue q;
    q.mpi_rank = rank;
    q.mpi_size = size;

    auto start_train = system_clock::now();

    // 各进程加载同一份模型，保证优先队列状态一致
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();

    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) *
                 microseconds::period::num /
                 microseconds::period::den;

    // 加载测试集
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

    q.init();

    if (rank == 0)
    {
        cout << "Starting MPI pipeline version, mpi_size = " << size << endl;
    }

    // 启动哈希线程
    thread hash_thread(hash_worker_thread, cref(test_set), ref(time_hash));

    const int BATCH_SIZE = size;        // PT 层面批量大小
    const int HASH_THRESHOLD = 1000000; // 每生成约 100 万候选后转交哈希线程
    const int MAX_GUESSES = 10000000;   // 生成总量上限

    int curr_num = 0;
    int history = 0;
    int next_report = 100000;

    size_t last_local_guess_size = 0;
    bool should_continue = true;

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
            auto start_guess = system_clock::now();

            // PT 层面并行：一次处理多个 PT
            q.PopNextBatch(BATCH_SIZE);

            auto end_guess = system_clock::now();
            auto duration_guess = duration_cast<microseconds>(end_guess - start_guess);
            time_guess += double(duration_guess.count()) *
                          microseconds::period::num /
                          microseconds::period::den;
        }

        // 只统计本轮新增候选数量，避免重复累计 q.guesses.size()
        int local_added = (int)(q.guesses.size() - last_local_guess_size);
        last_local_guess_size = q.guesses.size();

        int global_added = 0;

        MPI_Allreduce(&local_added, &global_added,
                      1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        curr_num += global_added;

        if (rank == 0 && history + curr_num >= next_report)
        {
            cout << "Guesses generated: " << history + curr_num << endl;
            next_report += 100000;
        }

        // 生成缓存达到阈值后，把本地 q.guesses 转交给哈希线程
        if (curr_num > HASH_THRESHOLD)
        {
            push_guesses_to_hash_queue(q.guesses);
            last_local_guess_size = 0;

            history += curr_num;
            curr_num = 0;
        }

        int should_exit = 0;

        if (rank == 0 && history + curr_num > MAX_GUESSES)
        {
            should_exit = 1;
        }

        MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (should_exit)
        {
            // 退出前，把还没有交给哈希线程的 guesses 也放进队列
            push_guesses_to_hash_queue(q.guesses);
            last_local_guess_size = 0;

            should_continue = false;
            break;
        }
    }

    // 通知哈希线程退出；哈希线程会先处理完 pending_hash_guesses 再结束
    hash_thread_should_exit = true;
    hash_thread.join();

    int cracked = local_cracked.load();
    int total_cracked = 0;

    int hashed = local_hashed.load();
    int total_hashed = 0;

    MPI_Reduce(&cracked, &total_cracked,
               1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&hashed, &total_hashed,
               1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // 并行程序的全局时间一般取所有进程中的最大值
    double global_guess_time = 0;
    double global_hash_time = 0;
    double global_train_time = 0;

    MPI_Reduce(&time_guess, &global_guess_time,
               1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&time_hash, &global_hash_time,
               1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&time_train, &global_train_time,
               1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "Guess time:" << global_guess_time << "seconds" << endl;
        cout << "Hash time:" << global_hash_time << "seconds" << endl;
        cout << "Train time:" << global_train_time << "seconds" << endl;
        cout << "Cracked:" << total_cracked << endl;
        cout << "Hashed:" << total_hashed << endl;
    }

    MPI_Finalize();
    return 0;
}