// top_k_selector.h
#ifndef TOP_K_SELECTOR_H
#define TOP_K_SELECTOR_H

#include <vector>
#include <utility>
#include <algorithm>
#include <functional>

namespace utils {

/**
 * A TopKSelector could be either online or offline.
 * Here we store distances and we report the k smallest distances.
 * This means that for some metrics (such as dot product) you want to
 * store negative distances to get the closest vector.
 *
 * We report the distances and ids of the top k items.
 * The id is just the timestamp (from 0 to number of items inserted so far)
 * of the item. You must be able to remap those timestamps to the original ids
 * if needed.
 * We adopt this strategy for two reasons:
 * - The value of k is small so we can afford the remapping;
 * - The number of distances to be checked is large so we want to save the
 *   time needed to create a vector of original ids and copy them.
 *
 * An online selector, such as an implementation of a Heap,
 * updates the data structure after every `push`.
 * The current top k values can be reported efficiently after every push.
 *
 * An offline selector (e.g., quickselect) may just collect every pushed
 * distances without doing anything. Then it can spend spends more time
 * (e.g., linear time) in computing the `topk` distances.
 *
 * An online selector may be faster if a lot of distance are processed
 * at once.
 */
class OnlineTopKSelector {
public:
    /**
     * Creates a new empty data structure to compute top-`k` distances.
     */
    OnlineTopKSelector(size_t k) : k_(k), next_id_(0) {}
    
    virtual ~OnlineTopKSelector() = default;

    /**
     * Pushes a new item `distance` with the current timestamp.
     * If the data structure has less than k distances, the current one is
     * inserted.
     * Otherwise, the current one replaces the largest distance
     * stored so far, if it is smaller.
     */
    virtual void push(float distance) {
        push_with_id(distance, next_id_++);
    }

    /**
     * Pushes a new item `distance` with a specified `id` as its timestamp.
     * If the data structure has less than k distances, the current one is
     * inserted.
     * Otherwise, the current one replaces the largest distance
     * stored so far, if it is smaller.
     */
    virtual void push_with_id(float distance, size_t id) = 0;

    /**
     * Pushes a slice of items `distances`.
     */
    virtual void extend(const std::vector<float>& distances) {
        for (const auto& distance : distances) {
            push(distance);
        }
    }

    /**
     * Returns the top-k distances and their timestamps.
     * The method returns these top-k distances as a
     * sorted (by decreasing distance) vector of pairs.
     */
    virtual std::vector<std::pair<float, size_t>> topk() const = 0;

    /**
     * Returns the number of distances stored in the data structure.
     */
    virtual size_t len() const { return k_; }

protected:
    size_t k_;
    size_t next_id_;
};

/**
 * Custom comparator for the heap that correctly handles negative values
 * We want to keep the k smallest values, so in a max-heap, we want to 
 * compare values such that the largest (which will be at the top of the heap)
 * is the one we'd remove first.
 */
struct HeapComparator {
    bool operator()(const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) const {
        return a.first < b.first; // For a max-heap, we want the largest value at the top
    }
};

/**
 * HeapFaiss implementation of OnlineTopKSelector
 * This implementation uses a max-heap to keep track of the k smallest distances.
 */
class HeapFaiss : public OnlineTopKSelector {
public:
    HeapFaiss(size_t k) : OnlineTopKSelector(k) {
        // Reserve space for k elements to avoid reallocations
        heap_.reserve(k);
    }

    void push_with_id(float distance, size_t id) override {
        if (heap_.size() < k_) {
            // If we have fewer than k elements, just add the new one
            heap_.emplace_back(distance, id);
            
            // If this was the k-th element, heapify the array
            if (heap_.size() == k_) {
                std::make_heap(heap_.begin(), heap_.end(), HeapComparator());
            }
        } else if (distance < heap_.front().first) {
            // If the new distance is smaller than the largest in our heap,
            // replace the largest element
            std::pop_heap(heap_.begin(), heap_.end(), HeapComparator());
            heap_.back() = std::make_pair(distance, id);
            std::push_heap(heap_.begin(), heap_.end(), HeapComparator());
        }
    }

    std::vector<std::pair<float, size_t>> topk() const override {
        // Create a copy of the heap
        auto result = heap_;
        
        // Sort the result in decreasing order of distance
        std::sort(result.begin(), result.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        return result;
    }

    size_t len() const override { return heap_.size(); }

private:
    std::vector<std::pair<float, size_t>> heap_;
};

} // namespace utils

#endif // TOP_K_SELECTOR_H
