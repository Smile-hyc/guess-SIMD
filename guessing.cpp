#include "PCFG.h"

using namespace std;

static const int MQ_FACTOR = 2;              // local queue 数量约为 2 * OpenMP 最大线程数
static const int MIN_OPENMP_GRAIN = 2048;    // 内部自动粒度控制，不需要命令行传阈值

PriorityQueue::~PriorityQueue()
{
    DestroyMultiQueue();
}

segment *PriorityQueue::LocateSegment(const segment &seg)
{
    if (seg.type == 1)
    {
        int id = m.FindLetter(seg);
        if (id >= 0)
            return &m.letters[id];
    }
    else if (seg.type == 2)
    {
        int id = m.FindDigit(seg);
        if (id >= 0)
            return &m.digits[id];
    }
    else if (seg.type == 3)
    {
        int id = m.FindSymbol(seg);
        if (id >= 0)
            return &m.symbols[id];
    }

    return nullptr;
}

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;

    int index = 0;
    for (int idx : pt.curr_indices)
    {
        segment *seg_data = LocateSegment(pt.content[index]);
        if (seg_data == nullptr)
        {
            index += 1;
            continue;
        }

        if (idx >= 0 && idx < (int)seg_data->ordered_freqs.size() && seg_data->total_freq > 0)
        {
            pt.prob *= seg_data->ordered_freqs[idx];
            pt.prob /= seg_data->total_freq;
        }

        index += 1;
    }
}

int PriorityQueue::RandomQueueIndex()
{
    // 简单快速的线性同余随机数，避免频繁调用 rand() 的额外开销。
    rng_state = rng_state * 1664525u + 1013904223u;
    return (mq_queue_count == 0) ? 0 : (int)(rng_state % mq_queue_count);
}

float PriorityQueue::QueueTopProb(int idx) const
{
    if (idx < 0 || idx >= (int)local_queues.size())
        return -1.0f;

    if (local_queues[idx].empty())
        return -1.0f;

    return local_queues[idx].top().prob;
}

int PriorityQueue::ChooseBetterQueue(int i, int j) const
{
    float pi = QueueTopProb(i);
    float pj = QueueTopProb(j);

    if (pi < 0 && pj < 0)
        return -1;
    if (pi >= pj)
        return i;
    return j;
}

void PriorityQueue::InitMultiQueue()
{
    DestroyMultiQueue();

    int max_threads = omp_get_max_threads();
    if (max_threads <= 0)
        max_threads = 1;

    mq_queue_count = max(4, max_threads * MQ_FACTOR);

    local_queues.clear();
    local_queues.resize(mq_queue_count);

    queue_locks.clear();
    queue_locks.resize(mq_queue_count);

    for (int i = 0; i < mq_queue_count; i++)
    {
        omp_init_lock(&queue_locks[i]);
    }

    locks_initialized = true;
    use_multiqueue = true;
    active_pt_count = 0;
    rng_state = 1234567u;
}

void PriorityQueue::DestroyMultiQueue()
{
    if (locks_initialized)
    {
        for (int i = 0; i < (int)queue_locks.size(); i++)
        {
            omp_destroy_lock(&queue_locks[i]);
        }
    }

    queue_locks.clear();
    local_queues.clear();

    locks_initialized = false;
    use_multiqueue = false;
    active_pt_count = 0;
    mq_queue_count = 0;
}

void PriorityQueue::InsertPT_MQ(const PT &pt)
{
    if (!use_multiqueue || mq_queue_count == 0)
    {
        priority.emplace_back(pt);
        return;
    }

    // MultiQueue 思路：随机找一个局部队列插入，不抢全局大队列。
    while (true)
    {
        int i = RandomQueueIndex();

        if (omp_test_lock(&queue_locks[i]))
        {
            local_queues[i].push(pt);
            active_pt_count += 1;
            omp_unset_lock(&queue_locks[i]);
            return;
        }
    }
}

bool PriorityQueue::PopBestPTByScan(PT &pt)
{
    int best_idx = -1;
    float best_prob = -1.0f;

    // 作为兜底策略：随机两选一连续失败时，扫描所有局部队列。
    // 这样可以避免队列快空时随机选不到非空队列。
    for (int i = 0; i < mq_queue_count; i++)
    {
        if (omp_test_lock(&queue_locks[i]))
        {
            if (!local_queues[i].empty())
            {
                float prob = local_queues[i].top().prob;
                if (best_idx == -1 || prob > best_prob)
                {
                    best_idx = i;
                    best_prob = prob;
                }
            }
            omp_unset_lock(&queue_locks[i]);
        }
    }

    if (best_idx == -1)
        return false;

    omp_set_lock(&queue_locks[best_idx]);

    if (local_queues[best_idx].empty())
    {
        omp_unset_lock(&queue_locks[best_idx]);
        return false;
    }

    pt = local_queues[best_idx].top();
    local_queues[best_idx].pop();
    active_pt_count -= 1;

    omp_unset_lock(&queue_locks[best_idx]);
    return true;
}

bool PriorityQueue::PopPT_MQ(PT &pt)
{
    if (!use_multiqueue || active_pt_count == 0)
        return false;

    // MultiQueue 论文中的核心：随机选两个局部队列，取队首概率更高的那个。
    // 这样不是严格全局最优，但比随机取一个队列更稳。
    int max_attempts = max(16, mq_queue_count * 2);

    for (int attempt = 0; attempt < max_attempts; attempt++)
    {
        int i = RandomQueueIndex();
        int j = RandomQueueIndex();

        int k = ChooseBetterQueue(i, j);
        if (k < 0)
            continue;

        if (omp_test_lock(&queue_locks[k]))
        {
            if (!local_queues[k].empty())
            {
                pt = local_queues[k].top();
                local_queues[k].pop();
                active_pt_count -= 1;

                omp_unset_lock(&queue_locks[k]);
                return true;
            }

            omp_unset_lock(&queue_locks[k]);
        }
    }

    return PopBestPTByScan(pt);
}

bool PriorityQueue::Empty() const
{
    if (use_multiqueue)
        return active_pt_count == 0;

    return priority.empty();
}

void PriorityQueue::init()
{
    InitMultiQueue();

    priority.clear();

    for (PT pt : m.ordered_pts)
    {
        pt.max_indices.clear();

        for (segment seg : pt.content)
        {
            segment *seg_data = LocateSegment(seg);
            if (seg_data != nullptr)
            {
                pt.max_indices.emplace_back((int)seg_data->ordered_values.size());
            }
            else
            {
                pt.max_indices.emplace_back(0);
            }
        }

        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;

        CalProb(pt);

        // 不再插入单一全局 priority，而是分散插入多个局部优先队列。
        InsertPT_MQ(pt);
    }
}

void PriorityQueue::PopNext()
{
    PT current;

    if (use_multiqueue)
    {
        if (!PopPT_MQ(current))
        {
            return;
        }

        Generate(current);

        vector<PT> new_pts = current.NewPTs();
        for (PT pt : new_pts)
        {
            CalProb(pt);
            InsertPT_MQ(pt);
        }

        return;
    }

    // 兼容旧逻辑：如果没有启用 MultiQueue，则使用原来的 priority。
    if (priority.empty())
        return;

    current = priority.front();

    Generate(current);

    vector<PT> new_pts = current.NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);

        bool inserted = false;
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    inserted = true;
                    break;
                }
            }

            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                inserted = true;
                break;
            }

            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                inserted = true;
                break;
            }
        }

        if (!inserted)
        {
            priority.emplace_back(pt);
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

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.empty())
        return;

    string prefix;
    int last_idx = (int)pt.content.size() - 1;

    // 多 segment 情况下，前面的 segment 仍然串行构造 prefix。
    // 这部分有顺序依赖，强行并行意义不大。
    if (pt.content.size() > 1)
    {
        for (int seg_idx = 0; seg_idx < last_idx; seg_idx++)
        {
            segment *seg_data = LocateSegment(pt.content[seg_idx]);
            if (seg_data == nullptr)
                continue;

            int value_idx = pt.curr_indices[seg_idx];
            if (value_idx >= 0 && value_idx < (int)seg_data->ordered_values.size())
            {
                prefix += seg_data->ordered_values[value_idx];
            }
        }
    }

    segment *last_seg_data = LocateSegment(pt.content[last_idx]);
    if (last_seg_data == nullptr)
        return;

    int loop_bound = 0;
    if (last_idx >= 0 && last_idx < (int)pt.max_indices.size())
    {
        loop_bound = min((int)last_seg_data->ordered_values.size(), pt.max_indices[last_idx]);
    }
    else
    {
        loop_bound = (int)last_seg_data->ordered_values.size();
    }

    if (loop_bound <= 0)
        return;

    // 候选编号直写：提前扩容，然后每个线程写自己的 guesses[base+i]。
    // 这样避免了 local vector 再 insert 合并，也避免了 critical。
    size_t base = guesses.size();
    guesses.resize(base + loop_bound);

    if (prefix.empty())
    {
#pragma omp parallel for schedule(static) if(loop_bound >= MIN_OPENMP_GRAIN)
        for (int i = 0; i < loop_bound; i++)
        {
            guesses[base + i] = last_seg_data->ordered_values[i];
        }
    }
    else
    {
#pragma omp parallel for schedule(static) if(loop_bound >= MIN_OPENMP_GRAIN)
        for (int i = 0; i < loop_bound; i++)
        {
            guesses[base + i] = prefix + last_seg_data->ordered_values[i];
        }
    }

    total_guesses = (int)guesses.size();
}