#include "PCFG.h"
#include <algorithm>

using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;

    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }

        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }

        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }

        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }

            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }

            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }

        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;

        CalProb(pt);

        priority.emplace_back(pt);
    }
}

void PriorityQueue::InsertNewPTs(const vector<PT> &new_pts)
{
    for (PT pt : new_pts)
    {
        if (priority.empty())
        {
            priority.emplace_back(pt);
            continue;
        }

        bool inserted = false;

        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                inserted = true;
                break;
            }

            if (iter != priority.end() - 1 && pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
            {
                priority.emplace(iter + 1, pt);
                inserted = true;
                break;
            }

            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                inserted = true;
                break;
            }
        }

        if (!inserted)
        {
            priority.emplace_back(pt);
        }
    }
}

void PriorityQueue::PopNext()
{
    GenerateMPI(priority.front());

    vector<PT> new_pts = priority.front().NewPTs();

    for (PT &pt : new_pts)
    {
        CalProb(pt);
    }

    InsertNewPTs(new_pts);

    priority.erase(priority.begin());
}

void PriorityQueue::PopNextBatch(int batch_size)
{
    if (priority.empty())
    {
        return;
    }

    int actual_batch_size = min(batch_size, (int)priority.size());

    // 先把本轮要处理的 PT 拷贝出来。
    // 所有进程的 priority 初始一致，所以这里拷出来的 batch_pts 也一致。
    vector<PT> batch_pts;
    batch_pts.reserve(actual_batch_size);

    for (int i = 0; i < actual_batch_size; i++)
    {
        batch_pts.emplace_back(priority[i]);
    }

    // PT 层面并行：
    // rank 0 处理 batch_pts[0]，rank 1 处理 batch_pts[1] ...
    // 如果 batch_size 大于 mpi_size，则按 i % mpi_size 继续分配。
    for (int i = 0; i < actual_batch_size; i++)
    {
        if (i % mpi_size == mpi_rank)
        {
            // 这里用 Generate，而不是 GenerateMPI。
            // 因为这个 PT 已经分给当前进程了，当前进程要生成这个 PT 的完整候选集合。
            Generate(batch_pts[i]);
        }
    }

    // 所有进程都用同样的 batch_pts 计算新 PT。
    // 这样可以避免复杂的 PT 序列化和广播，同时保证各进程队列状态一致。
    vector<PT> all_new_pts;

    for (int i = 0; i < actual_batch_size; i++)
    {
        vector<PT> new_pts = batch_pts[i].NewPTs();

        for (PT &pt : new_pts)
        {
            CalProb(pt);
            all_new_pts.emplace_back(pt);
        }
    }

    // 删除本轮已经处理的一批 PT
    priority.erase(priority.begin(), priority.begin() + actual_batch_size);

    // 按同样顺序插回所有新 PT
    InsertNewPTs(all_new_pts);
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;

        for (int i = pivot; i < (int)curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;

            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }

            curr_indices[i] -= 1;
        }

        pivot = init_pivot;
        return res;
    }

    return res;
}

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;

        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }

        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }

        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;

        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }

            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }

            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }

            seg_idx += 1;

            if (seg_idx == (int)pt.content.size() - 1)
            {
                break;
            }
        }

        segment *a = nullptr;
        int last_idx = pt.content.size() - 1;

        if (pt.content[last_idx].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[last_idx])];
        }

        if (pt.content[last_idx].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[last_idx])];
        }

        if (pt.content[last_idx].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[last_idx])];
        }

        for (int i = 0; i < pt.max_indices[last_idx]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}

void PriorityQueue::GenerateMPI(PT pt)
{
    CalProb(pt);

    if (mpi_size <= 1)
    {
        Generate(pt);
        return;
    }

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;

        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }

        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }

        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        int total_values = pt.max_indices[0];

        int values_per_process = total_values / mpi_size;
        int remainder = total_values % mpi_size;

        int start_idx = mpi_rank * values_per_process + min(mpi_rank, remainder);
        int end_idx = start_idx + values_per_process + (mpi_rank < remainder ? 1 : 0);

        for (int i = start_idx; i < end_idx; i++)
        {
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;

        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }

            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }

            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }

            seg_idx += 1;

            if (seg_idx == (int)pt.content.size() - 1)
            {
                break;
            }
        }

        segment *a = nullptr;
        int last_idx = pt.content.size() - 1;

        if (pt.content[last_idx].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[last_idx])];
        }

        if (pt.content[last_idx].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[last_idx])];
        }

        if (pt.content[last_idx].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[last_idx])];
        }

        int total_values = pt.max_indices[last_idx];

        int values_per_process = total_values / mpi_size;
        int remainder = total_values % mpi_size;

        int start_idx = mpi_rank * values_per_process + min(mpi_rank, remainder);
        int end_idx = start_idx + values_per_process + (mpi_rank < remainder ? 1 : 0);

        for (int i = start_idx; i < end_idx; i++)
        {
            string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}