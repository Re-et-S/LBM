#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <sched.h>
#include <pthread.h>

inline void set_self_affinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();    
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    
    if (rc != 0) {
        std::cerr << "Error pinning thread to core " << core_id << std::endl;
    }
}

class ThreadPool {
public:
    // Initialize pool with N threads
    explicit ThreadPool(size_t num_threads) : stop(false) {
        for(size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, i] {

                int core_id = (i + 1) % std::thread::hardware_concurrency();
                set_self_affinity(core_id);
                
                while(true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this]{ return stop || !tasks.empty(); });
                        
                        if(stop && tasks.empty()) return;
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    // Clean shutdown
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker : workers) {
            if(worker.joinable()) worker.join();
        }
    }

    // Splits the range [start, end) into chunks and waits for completion.
    void parallel_for(int start, int end, std::function<void(int, int)> func) {
        int range = end - start;
        if (range <= 0) return;

        int num_workers = workers.size();
        // Calculate chunk size (simple division)
        int chunk_size = (range + num_workers - 1) / num_workers;

        // Synchronization primitives for THIS BATCH only
        std::atomic<int> active_tasks(0);
        std::mutex wait_mutex;
        std::condition_variable wait_cv;

        // Launch tasks
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            for (int i = 0; i < num_workers; ++i) {
                int chunk_start = start + i * chunk_size;
                int chunk_end = std::min(start + (i + 1) * chunk_size, end);

                if (chunk_start >= chunk_end) break;

                active_tasks++; // Increment before pushing
                
                // Push lambda to queue
                tasks.emplace([chunk_start, chunk_end, &func, &active_tasks, &wait_cv, &wait_mutex]() {
                    // 1. Do the work
                    func(chunk_start, chunk_end);

                    // 2. Decrement and Notify if done
                    // fetch_sub returns the PREVIOUS value. If it was 1, it is now 0.
                    if (active_tasks.fetch_sub(1) == 1) {
                        std::unique_lock<std::mutex> lk(wait_mutex);
                        wait_cv.notify_one();
                    }
                });
            }
        }
        condition.notify_all(); // Wake up pool threads

        // Block here until all parts of the loop are done
        std::unique_lock<std::mutex> wait_lock(wait_mutex);
        wait_cv.wait(wait_lock, [&]{ return active_tasks.load() == 0; });
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};
