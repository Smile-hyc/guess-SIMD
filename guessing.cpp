#include "PCFG.h"
#include <algorithm>
#include <vector>
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
                pt.max_indices.emplace_back(
                    m.letters[m.FindLetter(seg)].ordered_values.size()
                );
            }

            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(
                    m.digits[m.FindDigit(seg)].ordered_values.size()
                );
            }

            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(
                    m.symbols[m.FindSymbol(seg)].ordered_values.size()
                );
            }
        }

        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;

        CalProb(pt);

        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    Generate(priority.front());

    vector<PT> new_pts = priority.front().NewPTs();

    for (PT pt : new_pts)
    {
        CalProb(pt);

        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }

            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }

            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    priority.erase(priority.begin());
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

        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
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

// OpenMP dynamic 调度版本：
// 小任务走串行，大任务进入 parallel for。
// 每个线程写自己的 local vector，parallel 结束后主线程统一合并。
static void generate_segment_openmp_dynamic(
    vector<string>& guesses,
    int& total_guesses,
    segment* seg_data,
    const string& prefix,
    int loop_bound,
    int thread_num,
    int threshold,
    int chunk_size)
{
    if (seg_data == nullptr)
    {
        return;
    }

    int actual_size = seg_data->ordered_values.size();

    if (loop_bound > actual_size)
    {
        loop_bound = actual_size;
    }

    if (loop_bound <= 0)
    {
        return;
    }

    if (thread_num <= 1 || loop_bound < threshold)
    {
        if (prefix.empty())
        {
            for (int i = 0; i < loop_bound; i += 1)
            {
                guesses.emplace_back(seg_data->ordered_values[i]);
            }
        }
        else
        {
            for (int i = 0; i < loop_bound; i += 1)
            {
                guesses.emplace_back(prefix + seg_data->ordered_values[i]);
            }
        }

        total_guesses += loop_bound;
        return;
    }

    if (thread_num > loop_bound)
    {
        thread_num = loop_bound;
    }

    if (chunk_size <= 0)
    {
        chunk_size = 1;
    }

    const vector<string>& values = seg_data->ordered_values;

    vector<vector<string>> all_local_guesses(thread_num);

#pragma omp parallel num_threads(thread_num)
    {
        int tid = omp_get_thread_num();

        vector<string> local_guesses;
        local_guesses.reserve(loop_bound / thread_num + chunk_size + 8);

        if (prefix.empty())
        {
#pragma omp for schedule(dynamic, chunk_size) nowait
            for (int i = 0; i < loop_bound; i += 1)
            {
                local_guesses.emplace_back(values[i]);
            }
        }
        else
        {
#pragma omp for schedule(dynamic, chunk_size) nowait
            for (int i = 0; i < loop_bound; i += 1)
            {
                local_guesses.emplace_back(prefix + values[i]);
            }
        }

        all_local_guesses[tid].swap(local_guesses);
    }

    for (int t = 0; t < thread_num; t += 1)
    {
        guesses.insert(
            guesses.end(),
            all_local_guesses[t].begin(),
            all_local_guesses[t].end()
        );
    }

    total_guesses += loop_bound;
}

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        if (pt.content.empty() || pt.max_indices.empty())
        {
            return;
        }

        segment* target_segment_data = nullptr;

        if (pt.content[0].type == 1)
        {
            target_segment_data = &m.letters[m.FindLetter(pt.content[0])];
        }
        else if (pt.content[0].type == 2)
        {
            target_segment_data = &m.digits[m.FindDigit(pt.content[0])];
        }
        else if (pt.content[0].type == 3)
        {
            target_segment_data = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        if (target_segment_data == nullptr)
        {
            return;
        }

        int loop_bound = pt.max_indices[0];

        generate_segment_openmp_dynamic(
            guesses,
            total_guesses,
            target_segment_data,
            "",
            loop_bound,
            omp_thread_num,
            omp_threshold,
            omp_chunk_size
        );
    }
    else
    {
        if (pt.content.empty() ||
            pt.curr_indices.empty() ||
            pt.max_indices.size() != pt.content.size())
        {
            return;
        }

        string prefix_guess_str;
        int seg_idx = 0;

        for (int idx : pt.curr_indices)
        {
            if (seg_idx >= pt.content.size() - 1)
            {
                break;
            }

            const segment& current_seg_template = pt.content[seg_idx];
            const segment* concrete_segment_data = nullptr;

            if (current_seg_template.type == 1)
            {
                concrete_segment_data = &m.letters[m.FindLetter(current_seg_template)];
            }
            else if (current_seg_template.type == 2)
            {
                concrete_segment_data = &m.digits[m.FindDigit(current_seg_template)];
            }
            else if (current_seg_template.type == 3)
            {
                concrete_segment_data = &m.symbols[m.FindSymbol(current_seg_template)];
            }

            if (concrete_segment_data &&
                idx >= 0 &&
                idx < concrete_segment_data->ordered_values.size())
            {
                prefix_guess_str += concrete_segment_data->ordered_values[idx];
            }
            else
            {
                return;
            }

            seg_idx += 1;
        }

        segment* last_segment_data = nullptr;
        int last_segment_idx = pt.content.size() - 1;

        if (last_segment_idx < 0)
        {
            return;
        }

        const segment& last_seg_template = pt.content[last_segment_idx];

        if (last_seg_template.type == 1)
        {
            last_segment_data = &m.letters[m.FindLetter(last_seg_template)];
        }
        else if (last_seg_template.type == 2)
        {
            last_segment_data = &m.digits[m.FindDigit(last_seg_template)];
        }
        else if (last_seg_template.type == 3)
        {
            last_segment_data = &m.symbols[m.FindSymbol(last_seg_template)];
        }

        if (last_segment_data == nullptr)
        {
            return;
        }

        int loop_bound = pt.max_indices[last_segment_idx];

        generate_segment_openmp_dynamic(
            guesses,
            total_guesses,
            last_segment_data,
            prefix_guess_str,
            loop_bound,
            omp_thread_num,
            omp_threshold,
            omp_chunk_size
        );
    }
}