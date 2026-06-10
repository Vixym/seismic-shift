// End-to-end verification of the dynamic update path (insert / delete / resize).
//
// Loads a previously built (dynamic-support) index, then:
//   1. inserts a batch of documents and verifies each becomes retrievable
//   2. deletes a batch of documents and verifies they disappear after resize()
//   3. reports per-operation latency
//
// This exercises MyInvertedIndex::{insert_doc, delete_doc, resize}, which were
// never run by the existing perf binary (the benchmark there is commented out).

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <optional>

#include "../inverted_index.h"
#include "../sparse_dataset.h"

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/optional.hpp>

#include "../my_inverted_index.h"

using namespace seismic;
using TIndex = MyInvertedIndex<float>;
using TDataset = SparseDatasetMut<float>;
using Clock = std::chrono::high_resolution_clock;

static TIndex load_index(const std::string& path) {
    std::string full = path + ".index.seismic";
    std::cout << "Loading index from " << full << " ..." << std::endl;
    std::ifstream file(full, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + full);
    TIndex index;
    cereal::BinaryInputArchive archive(file);
    archive(index);
    std::cout << "Loaded. documents=" << index.len() << " dim=" << index.dim() << std::endl;
    return index;
}

// Does doc_id appear among the search results for the given query vector?
static bool retrievable(const TIndex& index,
                        const std::vector<uint16_t>& comps,
                        const std::vector<float>& vals,
                        size_t doc_id, size_t k) {
    auto res = index.search(comps, vals, k, /*query_cut*/ comps.size(),
                            /*heap_factor*/ 0.7f, /*n_knn*/ 0, /*first_sorted*/ false);
    for (auto& [score, id] : res) if (id == doc_id) return true;
    return false;
}

int main(int argc, char* argv[]) {
    std::string index_file, query_file;
    size_t n_ops = 200;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--index-file" && i + 1 < argc) index_file = argv[++i];
        else if (a == "--query-file" && i + 1 < argc) query_file = argv[++i];
        else if (a == "--n-ops" && i + 1 < argc) n_ops = std::stoul(argv[++i]);
    }
    if (index_file.empty() || query_file.empty()) {
        std::cerr << "usage: test_dynamic --index-file F --query-file F [--n-ops N]\n";
        return 1;
    }

    try {
        TIndex index = load_index(index_file);
        TDataset queries = TDataset::read_bin_file(query_file);
        std::cout << "Loaded " << queries.len() << " query vectors (used as docs to insert)\n";
        n_ops = std::min(n_ops, queries.len());

        // ---- baseline search sanity ----
        {
            auto q = queries.get(0);
            auto res = index.search(q.first, q.second, 10, q.first.size(), 0.7f, 0, false);
            std::cout << "\n[baseline] search returned " << res.size() << " results"
                      << " (top score " << (res.empty() ? 0.f : res.front().first) << ")\n";
            if (res.size() != 10) { std::cerr << "FAIL: baseline search did not return k=10\n"; return 2; }
        }

        // ---- INSERT ----
        size_t n_before = index.len();
        std::cout << "\n[insert] inserting " << n_ops << " docs ...\n";
        auto t0 = Clock::now();
        for (size_t i = 0; i < n_ops; ++i) {
            auto q = queries.get(i);
            index.insert_doc(q.first, q.second);
        }
        auto t1 = Clock::now();
        size_t n_after = index.len();

        if (n_after != n_before + n_ops) {
            std::cerr << "FAIL: len after insert = " << n_after
                      << ", expected " << (n_before + n_ops) << "\n";
            return 3;
        }
        // Correctness: each freshly inserted doc (== its query vector) must be retrievable.
        size_t ok = 0;
        for (size_t i = 0; i < n_ops; ++i) {
            auto q = queries.get(i);
            if (retrievable(index, q.first, q.second, n_before + i, 10)) ++ok;
        }
        double ins_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / double(n_ops);
        std::cout << "[insert] len " << n_before << " -> " << n_after
                  << " | retrievable " << ok << "/" << n_ops
                  << " | " << ins_us << " us/insert\n";
        if (ok != n_ops) std::cerr << "WARN: " << (n_ops - ok)
                                   << " inserted docs were not retrievable by their own vector\n";

        // ---- DELETE ----
        std::cout << "\n[delete] deleting the " << n_ops << " just-inserted docs ...\n";
        auto t2 = Clock::now();
        for (size_t i = 0; i < n_ops; ++i) index.delete_doc(n_before + i);
        auto t3 = Clock::now();
        index.resize();
        auto t4 = Clock::now();

        // Correctness: deleted docs must no longer be retrievable by their own vector.
        size_t gone = 0;
        for (size_t i = 0; i < n_ops; ++i) {
            auto q = queries.get(i);
            if (!retrievable(index, q.first, q.second, n_before + i, 10)) ++gone;
        }
        double del_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / double(n_ops);
        double resize_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
        std::cout << "[delete] removed " << n_ops << " docs | gone " << gone << "/" << n_ops
                  << " | " << del_us << " us/delete | resize " << resize_ms << " ms\n";

        // ---- final sanity ----
        {
            auto q = queries.get(0);
            auto res = index.search(q.first, q.second, 10, q.first.size(), 0.7f, 0, false);
            std::cout << "\n[final] search returned " << res.size() << " results after insert+delete+resize\n";
            if (res.empty()) { std::cerr << "FAIL: search broken after dynamic ops\n"; return 4; }
        }

        bool pass = (ok == n_ops) && (gone == n_ops);
        std::cout << "\n==== DYNAMIC VERIFICATION " << (pass ? "PASSED" : "COMPLETED WITH WARNINGS") << " ====\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
