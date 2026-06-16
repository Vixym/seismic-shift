// Dynamic-workload benchmark: does the centroid+radius index preserve pruning
// quality (recall + latency) under churn?
//
// Loads a centroid+radius index, then repeatedly deletes a batch of random
// (non-protected) documents and re-runs the full query set with the radius bound
// active (alpha>0), writing per-stage result files for offline RR@10. Finally it
// runs resize() (compaction, which recomputes tight radii) and re-measures.
//
// Protected doc-ids (the qrels-relevant documents) are never deleted, so any recall
// change reflects pruning degradation rather than removed answers.

#include "../inverted_index.h"
#include "../sparse_dataset.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/optional.hpp>
#include "../my_inverted_index.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <random>
#include <chrono>

using namespace seismic;
using TIndex = MyInvertedIndex<float>;
using TDataset = SparseDatasetMut<float>;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    std::string index_file, query_file, protected_file, out_prefix = "/tmp/churn";
    float alpha = 0.15f, heap_factor = 0.9f;
    size_t query_cut = 3, deletes_per_round = 500000, n_rounds = 4, k = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--index-file" && i+1<argc) index_file = argv[++i];
        else if (a == "--query-file" && i+1<argc) query_file = argv[++i];
        else if (a == "--protected-file" && i+1<argc) protected_file = argv[++i];
        else if (a == "--out-prefix" && i+1<argc) out_prefix = argv[++i];
        else if (a == "--alpha" && i+1<argc) alpha = std::stof(argv[++i]);
        else if (a == "--query-cut" && i+1<argc) query_cut = std::stoul(argv[++i]);
        else if (a == "--heap-factor" && i+1<argc) heap_factor = std::stof(argv[++i]);
        else if (a == "--deletes-per-round" && i+1<argc) deletes_per_round = std::stoul(argv[++i]);
        else if (a == "--n-rounds" && i+1<argc) n_rounds = std::stoul(argv[++i]);
    }
    if (index_file.empty() || query_file.empty()) { std::cerr << "need --index-file and --query-file\n"; return 1; }

    std::cout << "Loading index " << index_file << " ...\n";
    TIndex index;
    { std::ifstream f(index_file + ".index.seismic", std::ios::binary);
      if (!f) { std::cerr << "cannot open index\n"; return 1; }
      cereal::BinaryInputArchive ar(f); ar(index); }
    const size_t n0 = index.len();
    std::cout << "docs=" << n0 << " dim=" << index.dim() << " alpha=" << alpha
              << " cut=" << query_cut << " hf=" << heap_factor << "\n";

    TDataset queries = TDataset::read_bin_file(query_file);
    const size_t nq = queries.len();

    std::unordered_set<size_t> protecteds;
    if (!protected_file.empty()) {
        std::ifstream pf(protected_file); size_t id;
        while (pf >> id) protecteds.insert(id);
        std::cout << "protected doc-ids: " << protecteds.size() << "\n";
    }

    auto run_queries = [&](const std::string& stage) {
        std::ofstream out(out_prefix + "_" + stage);
        auto t0 = Clock::now();
        for (size_t qi = 0; qi < nq; ++qi) {
            auto [qc, qv] = queries.get(qi);
            auto res = index.search(qc, qv, k, query_cut, heap_factor, /*n_knn*/0,
                                    /*first_sorted*/false, alpha, /*debug*/false);
            for (size_t r = 0; r < res.size(); ++r)
                out << qi << "\t" << res[r].second << "\t" << (r+1) << "\t" << res[r].first << "\n";
        }
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-t0).count();
        std::cout << "[stage " << stage << "] alive=" << index.len()
                  << "  " << (double(us)/nq) << " us/query  -> " << (out_prefix + "_" + stage) << "\n";
    };

    // Stage 0: baseline (no churn)
    run_queries("0");

    // Deterministic RNG (seed fixed for reproducibility).
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<size_t> pick(0, n0 - 1);

    size_t total_deleted = 0;
    for (size_t round = 1; round <= n_rounds; ++round) {
        size_t done = 0, attempts = 0, cap = deletes_per_round * 20;
        auto td = Clock::now();
        while (done < deletes_per_round && attempts < cap) {
            ++attempts;
            size_t id = pick(rng);
            if (protecteds.count(id)) continue;
            if (!index.dataset().is_alive(id)) continue;
            index.delete_doc(id);
            ++done;
        }
        total_deleted += done;
        auto del_us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-td).count();
        std::cout << "[round " << round << "] deleted " << done << " (total " << total_deleted
                  << ", " << (100.0*total_deleted/n0) << "% of collection)  "
                  << (double(del_us)/std::max<size_t>(done,1)) << " us/delete\n";
        run_queries(std::to_string(round));
    }

    // Compaction: should recompute tight radii and restore pruning.
    std::cout << "resize() ...\n";
    auto tr = Clock::now();
    index.resize();
    auto rs_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-tr).count();
    std::cout << "resize took " << rs_ms << " ms\n";
    run_queries("resize");

    return 0;
}
