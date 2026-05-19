#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <vector>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

/*
 * 改进思路：
 * 1. 原始 train() 是逐行读入口令，然后直接调用 parse(pw) 修改全局 model。
 *    这样会导致大量 FindPT / FindLetter / FindDigit / FindSymbol 线性查找。
 *
 * 2. 新版本把训练阶段看成“大规模计数问题”：
 *      每个线程先维护自己的 local counter；
 *      线程之间不共享写 model；
 *      最后再把所有 local counter 归并成全局统计结果。
 *
 * 3. 训练结束后，根据全局统计结果一次性构造：
 *      preterminals / preterm_freq
 *      letters / letters_freq
 *      digits / digits_freq
 *      symbols / symbols_freq
 *
 * 4. 为了兼容原来的 guessing.cpp，仍然保留 FindPT / FindLetter / FindDigit / FindSymbol 接口。
 */


// ========== 一些辅助 key 编码函数 ==========

static string MakePTKey(const vector<segment> &segs)
{
    string key;
    key.reserve(segs.size() * 6);

    for (const auto &seg : segs)
    {
        key += to_string(seg.type);
        key += ":";
        key += to_string(seg.length);
        key += "|";
    }

    return key;
}

static string MakePTKey(const PT &pt)
{
    return MakePTKey(pt.content);
}

static vector<segment> DecodePTKey(const string &key)
{
    vector<segment> content;

    size_t pos = 0;
    while (pos < key.size())
    {
        size_t colon = key.find(':', pos);
        size_t bar = key.find('|', pos);

        if (colon == string::npos || bar == string::npos)
        {
            break;
        }

        int type = stoi(key.substr(pos, colon - pos));
        int length = stoi(key.substr(colon + 1, bar - colon - 1));

        content.emplace_back(type, length);
        pos = bar + 1;
    }

    return content;
}


// ========== 给 model 建快速索引，避免 FindXX 每次线性扫描 ==========

static unordered_map<const model *, unordered_map<string, int>> g_pt_index;
static unordered_map<const model *, unordered_map<int, int>> g_letter_index;
static unordered_map<const model *, unordered_map<int, int>> g_digit_index;
static unordered_map<const model *, unordered_map<int, int>> g_symbol_index;

static void RebuildModelIndex(const model *m)
{
    auto &pt_index = g_pt_index[m];
    auto &letter_index = g_letter_index[m];
    auto &digit_index = g_digit_index[m];
    auto &symbol_index = g_symbol_index[m];

    pt_index.clear();
    letter_index.clear();
    digit_index.clear();
    symbol_index.clear();

    for (int i = 0; i < (int)m->preterminals.size(); i++)
    {
        pt_index[MakePTKey(m->preterminals[i])] = i;
    }

    for (int i = 0; i < (int)m->letters.size(); i++)
    {
        letter_index[m->letters[i].length] = i;
    }

    for (int i = 0; i < (int)m->digits.size(); i++)
    {
        digit_index[m->digits[i].length] = i;
    }

    for (int i = 0; i < (int)m->symbols.size(); i++)
    {
        symbol_index[m->symbols[i].length] = i;
    }
}


// ========== 线程局部统计结构 ==========

struct LocalTrainCounter
{
    int total_preterm = 0;

    unordered_map<string, int> pt_freq;

    unordered_map<int, unordered_map<string, int>> letters;
    unordered_map<int, unordered_map<string, int>> digits;
    unordered_map<int, unordered_map<string, int>> symbols;
};

static void AddSegmentToLocal(LocalTrainCounter &local, int type, const string &value)
{
    int len = (int)value.length();

    if (type == 1)
    {
        local.letters[len][value] += 1;
    }
    else if (type == 2)
    {
        local.digits[len][value] += 1;
    }
    else
    {
        local.symbols[len][value] += 1;
    }
}

static void ParseToLocalCounter(const string &pw, LocalTrainCounter &local)
{
    vector<segment> pt_content;

    string curr_part = "";
    int curr_type = 0;

    for (char ch : pw)
    {
        int new_type;

        if (isalpha((unsigned char)ch))
        {
            new_type = 1;
        }
        else if (isdigit((unsigned char)ch))
        {
            new_type = 2;
        }
        else
        {
            new_type = 3;
        }

        if (curr_type == 0)
        {
            curr_type = new_type;
            curr_part += ch;
        }
        else if (new_type == curr_type)
        {
            curr_part += ch;
        }
        else
        {
            AddSegmentToLocal(local, curr_type, curr_part);
            pt_content.emplace_back(curr_type, (int)curr_part.length());

            curr_part.clear();
            curr_type = new_type;
            curr_part += ch;
        }
    }

    if (!curr_part.empty())
    {
        AddSegmentToLocal(local, curr_type, curr_part);
        pt_content.emplace_back(curr_type, (int)curr_part.length());
    }

    if (!pt_content.empty())
    {
        string pt_key = MakePTKey(pt_content);
        local.pt_freq[pt_key] += 1;
        local.total_preterm += 1;
    }
}

static void MergeNestedCounter(
    unordered_map<int, unordered_map<string, int>> &global_counter,
    const unordered_map<int, unordered_map<string, int>> &local_counter)
{
    for (const auto &len_pair : local_counter)
    {
        int len = len_pair.first;
        const auto &value_map = len_pair.second;

        auto &dst = global_counter[len];

        for (const auto &value_pair : value_map)
        {
            dst[value_pair.first] += value_pair.second;
        }
    }
}

static void BuildSegmentsFromCounter(
    vector<segment> &dst_segments,
    unordered_map<int, int> &dst_freq,
    unordered_map<int, unordered_map<string, int>> &counter,
    int type)
{
    vector<int> lengths;
    lengths.reserve(counter.size());

    for (const auto &p : counter)
    {
        lengths.emplace_back(p.first);
    }

    sort(lengths.begin(), lengths.end());

    dst_segments.clear();
    dst_freq.clear();

    int id = 0;

    for (int len : lengths)
    {
        segment seg(type, len);

        int total = 0;
        int value_id = 0;

        for (const auto &value_pair : counter[len])
        {
            const string &value = value_pair.first;
            int freq = value_pair.second;

            seg.values[value] = value_id;
            seg.freqs[value_id] = freq;

            value_id += 1;
            total += freq;
        }

        dst_segments.emplace_back(seg);
        dst_freq[id] = total;

        id += 1;
    }
}


// ========== model::train：并行训练入口 ==========

void model::train(string path)
{
    string pw;
    ifstream train_set(path);

    vector<string> passwords;
    passwords.reserve(3000000);

    int lines = 0;

    cout << "Training..." << endl;
    cout << "Training phase 1: reading passwords..." << endl;

    while (train_set >> pw)
    {
        lines += 1;
        passwords.emplace_back(pw);

        if (lines % 10000 == 0)
        {
            cout << "Lines processed: " << lines << endl;

            if (lines > 3000000)
            {
                break;
            }
        }
    }

    cout << "Training phase 1 done. Total lines: " << passwords.size() << endl;
    cout << "Training phase 2: parallel parsing and local counting..." << endl;

    int thread_count = 1;

#ifdef _OPENMP
    thread_count = omp_get_max_threads();
    if (thread_count <= 0)
    {
        thread_count = 1;
    }
#endif

    vector<LocalTrainCounter> locals(thread_count);

#ifdef _OPENMP
#pragma omp parallel
    {
        int tid = omp_get_thread_num();

#pragma omp for schedule(static)
        for (int i = 0; i < (int)passwords.size(); i++)
        {
            ParseToLocalCounter(passwords[i], locals[tid]);
        }
    }
#else
    for (int i = 0; i < (int)passwords.size(); i++)
    {
        ParseToLocalCounter(passwords[i], locals[0]);
    }
#endif

    cout << "Training phase 3: merging local counters..." << endl;

    unordered_map<string, int> global_pt_freq;
    unordered_map<int, unordered_map<string, int>> global_letters;
    unordered_map<int, unordered_map<string, int>> global_digits;
    unordered_map<int, unordered_map<string, int>> global_symbols;

    total_preterm = 0;

    for (const auto &local : locals)
    {
        total_preterm += local.total_preterm;

        for (const auto &pt_pair : local.pt_freq)
        {
            global_pt_freq[pt_pair.first] += pt_pair.second;
        }

        MergeNestedCounter(global_letters, local.letters);
        MergeNestedCounter(global_digits, local.digits);
        MergeNestedCounter(global_symbols, local.symbols);
    }

    cout << "Training phase 4: building model structures..." << endl;

    preterm_id = -1;
    letters_id = -1;
    digits_id = -1;
    symbols_id = -1;

    preterminals.clear();
    letters.clear();
    digits.clear();
    symbols.clear();

    preterm_freq.clear();
    letters_freq.clear();
    digits_freq.clear();
    symbols_freq.clear();

    ordered_pts.clear();

    BuildSegmentsFromCounter(letters, letters_freq, global_letters, 1);
    BuildSegmentsFromCounter(digits, digits_freq, global_digits, 2);
    BuildSegmentsFromCounter(symbols, symbols_freq, global_symbols, 3);

    letters_id = (int)letters.size() - 1;
    digits_id = (int)digits.size() - 1;
    symbols_id = (int)symbols.size() - 1;

    vector<string> pt_keys;
    pt_keys.reserve(global_pt_freq.size());

    for (const auto &p : global_pt_freq)
    {
        pt_keys.emplace_back(p.first);
    }

    sort(pt_keys.begin(), pt_keys.end());

    int pt_id = 0;

    for (const string &key : pt_keys)
    {
        PT pt;
        pt.content = DecodePTKey(key);

        pt.curr_indices.clear();
        for (int i = 0; i < (int)pt.content.size(); i++)
        {
            pt.curr_indices.emplace_back(0);
        }

        preterminals.emplace_back(pt);
        preterm_freq[pt_id] = global_pt_freq[key];

        pt_id += 1;
    }

    preterm_id = (int)preterminals.size() - 1;

    RebuildModelIndex(this);

    cout << "Training build done. Total PTs: " << preterminals.size() << endl;
}


// ========== 查找函数：优先使用哈希索引 ==========

int model::FindPT(PT pt)
{
    auto it_model = g_pt_index.find(this);
    if (it_model != g_pt_index.end())
    {
        auto it = it_model->second.find(MakePTKey(pt));
        if (it != it_model->second.end())
        {
            return it->second;
        }
    }

    for (int id = 0; id < (int)preterminals.size(); id += 1)
    {
        if (preterminals[id].content.size() != pt.content.size())
        {
            continue;
        }

        bool equal_flag = true;

        for (int idx = 0; idx < (int)preterminals[id].content.size(); idx += 1)
        {
            if (preterminals[id].content[idx].type != pt.content[idx].type ||
                preterminals[id].content[idx].length != pt.content[idx].length)
            {
                equal_flag = false;
                break;
            }
        }

        if (equal_flag)
        {
            return id;
        }
    }

    return -1;
}

int model::FindLetter(segment seg)
{
    auto it_model = g_letter_index.find(this);
    if (it_model != g_letter_index.end())
    {
        auto it = it_model->second.find(seg.length);
        if (it != it_model->second.end())
        {
            return it->second;
        }
    }

    for (int id = 0; id < (int)letters.size(); id += 1)
    {
        if (letters[id].length == seg.length)
        {
            return id;
        }
    }

    return -1;
}

int model::FindDigit(segment seg)
{
    auto it_model = g_digit_index.find(this);
    if (it_model != g_digit_index.end())
    {
        auto it = it_model->second.find(seg.length);
        if (it != it_model->second.end())
        {
            return it->second;
        }
    }

    for (int id = 0; id < (int)digits.size(); id += 1)
    {
        if (digits[id].length == seg.length)
        {
            return id;
        }
    }

    return -1;
}

int model::FindSymbol(segment seg)
{
    auto it_model = g_symbol_index.find(this);
    if (it_model != g_symbol_index.end())
    {
        auto it = it_model->second.find(seg.length);
        if (it != it_model->second.end())
        {
            return it->second;
        }
    }

    for (int id = 0; id < (int)symbols.size(); id += 1)
    {
        if (symbols[id].length == seg.length)
        {
            return id;
        }
    }

    return -1;
}


// ========== 原有基础函数 ==========

void PT::insert(segment seg)
{
    content.emplace_back(seg);
}

void segment::insert(string value)
{
    if (values.find(value) == values.end())
    {
        values[value] = values.size();
        freqs[values[value]] = 1;
    }
    else
    {
        freqs[values[value]] += 1;
    }
}

void segment::order()
{
    ordered_values.clear();
    ordered_freqs.clear();
    total_freq = 0;

    for (pair<string, int> value : values)
    {
        ordered_values.emplace_back(value.first);
    }

    sort(ordered_values.begin(), ordered_values.end(),
         [this](const string &a, const string &b)
         {
             int fa = freqs.at(values[a]);
             int fb = freqs.at(values[b]);

             if (fa != fb)
             {
                 return fa > fb;
             }

             return a < b;
         });

    for (const string &val : ordered_values)
    {
        int freq = freqs.at(values[val]);
        ordered_freqs.emplace_back(freq);
        total_freq += freq;
    }
}


// ========== 保留 parse：用于兼容旧代码或单独调用 ==========

void model::parse(string pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0;

    auto flush_segment = [&](int type, const string &part)
    {
        if (part.empty())
        {
            return;
        }

        segment seg(type, (int)part.length());

        if (type == 1)
        {
            int id = FindLetter(seg);
            if (id == -1)
            {
                id = GetNextLettersID();
                letters.emplace_back(seg);
                letters_freq[id] = 1;
                letters[id].insert(part);
                g_letter_index[this][seg.length] = id;
            }
            else
            {
                letters_freq[id] += 1;
                letters[id].insert(part);
            }
        }
        else if (type == 2)
        {
            int id = FindDigit(seg);
            if (id == -1)
            {
                id = GetNextDigitsID();
                digits.emplace_back(seg);
                digits_freq[id] = 1;
                digits[id].insert(part);
                g_digit_index[this][seg.length] = id;
            }
            else
            {
                digits_freq[id] += 1;
                digits[id].insert(part);
            }
        }
        else
        {
            int id = FindSymbol(seg);
            if (id == -1)
            {
                id = GetNextSymbolsID();
                symbols.emplace_back(seg);
                symbols_freq[id] = 1;
                symbols[id].insert(part);
                g_symbol_index[this][seg.length] = id;
            }
            else
            {
                symbols_freq[id] += 1;
                symbols[id].insert(part);
            }
        }

        pt.insert(seg);
    };

    for (char ch : pw)
    {
        int new_type;

        if (isalpha((unsigned char)ch))
        {
            new_type = 1;
        }
        else if (isdigit((unsigned char)ch))
        {
            new_type = 2;
        }
        else
        {
            new_type = 3;
        }

        if (curr_type == 0)
        {
            curr_type = new_type;
            curr_part += ch;
        }
        else if (new_type == curr_type)
        {
            curr_part += ch;
        }
        else
        {
            flush_segment(curr_type, curr_part);
            curr_part.clear();
            curr_type = new_type;
            curr_part += ch;
        }
    }

    if (!curr_part.empty())
    {
        flush_segment(curr_type, curr_part);
    }

    total_preterm += 1;

    int pt_id = FindPT(pt);

    if (pt_id == -1)
    {
        for (int i = 0; i < (int)pt.content.size(); i += 1)
        {
            pt.curr_indices.emplace_back(0);
        }

        pt_id = GetNextPretermID();
        preterminals.emplace_back(pt);
        preterm_freq[pt_id] = 1;
        g_pt_index[this][MakePTKey(pt)] = pt_id;
    }
    else
    {
        preterm_freq[pt_id] += 1;
    }
}


// ========== 打印函数 ==========

void segment::PrintSeg()
{
    if (type == 1)
    {
        cout << "L" << length;
    }
    if (type == 2)
    {
        cout << "D" << length;
    }
    if (type == 3)
    {
        cout << "S" << length;
    }
}

void segment::PrintValues()
{
    for (string iter : ordered_values)
    {
        cout << iter << " freq:" << freqs[values[iter]] << endl;
    }
}

void PT::PrintPT()
{
    for (auto iter : content)
    {
        iter.PrintSeg();
    }
}

void model::print()
{
    cout << "preterminals:" << endl;

    for (int i = 0; i < (int)preterminals.size(); i += 1)
    {
        preterminals[i].PrintPT();
        cout << " freq:" << preterm_freq[i];
        cout << endl;
    }

    for (auto iter : ordered_pts)
    {
        iter.PrintPT();
        cout << " freq:" << preterm_freq[FindPT(iter)];
        cout << endl;
    }

    cout << "segments:" << endl;

    for (int i = 0; i < (int)letters.size(); i += 1)
    {
        letters[i].PrintSeg();
        cout << " freq:" << letters_freq[i];
        cout << endl;
    }

    for (int i = 0; i < (int)digits.size(); i += 1)
    {
        digits[i].PrintSeg();
        cout << " freq:" << digits_freq[i];
        cout << endl;
    }

    for (int i = 0; i < (int)symbols.size(); i += 1)
    {
        symbols[i].PrintSeg();
        cout << " freq:" << symbols_freq[i];
        cout << endl;
    }
}

bool compareByPretermProb(const PT &a, const PT &b)
{
    if (a.preterm_prob != b.preterm_prob)
    {
        return a.preterm_prob > b.preterm_prob;
    }

    return MakePTKey(a) < MakePTKey(b);
}


// ========== order：使用已有 preterm_freq，不再重复线性查找 ==========

void model::order()
{
    cout << "Training phase 5: Ordering segment values and PTs..." << endl;

    ordered_pts.clear();

    for (int i = 0; i < (int)preterminals.size(); i++)
    {
        PT pt = preterminals[i];

        if (total_preterm > 0)
        {
            pt.preterm_prob = float(preterm_freq[i]) / total_preterm;
        }
        else
        {
            pt.preterm_prob = 0;
        }

        ordered_pts.emplace_back(pt);
    }

    cout << "total pts" << ordered_pts.size() << endl;

    sort(ordered_pts.begin(), ordered_pts.end(), compareByPretermProb);

    cout << "Ordering letters" << endl;
    for (int i = 0; i < (int)letters.size(); i += 1)
    {
        letters[i].order();
    }

    cout << "Ordering digits" << endl;
    for (int i = 0; i < (int)digits.size(); i += 1)
    {
        digits[i].order();
    }

    cout << "ordering symbols" << endl;
    for (int i = 0; i < (int)symbols.size(); i += 1)
    {
        symbols[i].order();
    }

    RebuildModelIndex(this);
}