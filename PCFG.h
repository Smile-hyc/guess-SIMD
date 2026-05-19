#ifndef PCFG_H
#define PCFG_H

#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>

using namespace std;

class segment
{
public:
    int type;   // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如 S6 的长度就是 6

    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    }

    void PrintSeg();

    vector<string> ordered_values;
    vector<int> ordered_freqs;

    int total_freq = 0;

    unordered_map<string, int> values;
    unordered_map<int, int> freqs;

    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    vector<segment> content;

    int pivot = 0;

    void insert(segment seg);
    void PrintPT();

    vector<PT> NewPTs();

    vector<int> curr_indices;
    vector<int> max_indices;

    float preterm_prob = 0;
    float prob = 0;
};

class model
{
public:
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;

    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    }

    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    }

    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    }

    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    }

    int total_preterm = 0;

    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;

    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    void train(string train_path);
    void store(string store_path);
    void load(string load_path);
    void parse(string pw);
    void order();
    void print();
};

struct PTCompare
{
    // priority_queue 默认把“比较结果为 false 的更大元素”放到 top。
    // 这里让 prob 更大的 PT 优先弹出。
    bool operator()(const PT &a, const PT &b) const
    {
        return a.prob < b.prob;
    }
};

class PriorityQueue
{
public:
    // 保留原来的 priority，兼容旧代码；新版本主要使用 local_queues。
    vector<PT> priority;

    model m;

    int total_guesses = 0;
    vector<string> guesses;

    // MultiQueue 风格的多个局部优先队列
    vector<priority_queue<PT, vector<PT>, PTCompare>> local_queues;
    vector<omp_lock_t> queue_locks;

    int mq_queue_count = 0;
    bool use_multiqueue = false;
    bool locks_initialized = false;
    size_t active_pt_count = 0;

    unsigned int rng_state = 1234567u;

    PriorityQueue() = default;
    ~PriorityQueue();

    void CalProb(PT &pt);
    void init();

    void Generate(PT pt);
    void PopNext();

    bool Empty() const;

    // MultiQueue 相关操作
    void InitMultiQueue();
    void DestroyMultiQueue();

    void InsertPT_MQ(const PT &pt);
    bool PopPT_MQ(PT &pt);

    // 辅助函数
    segment *LocateSegment(const segment &seg);
    int RandomQueueIndex();
    float QueueTopProb(int idx) const;
    int ChooseBetterQueue(int i, int j) const;
    bool PopBestPTByScan(PT &pt);
};

#endif