#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <vector>

using namespace std;

class segment
{
public:
    int type;   // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6

    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

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

    float preterm_prob;
    float prob;
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
    };

    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };

    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };

    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

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

class PriorityQueue
{
public:
    vector<PT> priority;

    model m;

    void CalProb(PT &pt);
    void init();

    // MPI 进程信息
    int mpi_rank = 0;
    int mpi_size = 1;

    // 基础 MPI 版本：单个 PT 内部按 rank 分片
    void GenerateMPI(PT pt);

    // 串行生成：用于 PT 层面并行时，某个 rank 处理完整 PT
    void Generate(PT pt);

    // 基础 MPI 版本：每次处理一个 PT
    void PopNext();

    // PT 层面并行版本：一次取出多个 PT
    void PopNextBatch(int batch_size = 4);

    // 将新 PT 按概率插回优先队列
    void InsertNewPTs(const vector<PT> &new_pts);

    int total_guesses = 0;
    vector<string> guesses;
};