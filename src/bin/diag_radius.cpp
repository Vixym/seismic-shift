// Diagnostic: how tight is the "centroid + residual radius" upper bound on each block?
//
// For a sample of blocks it recomputes, in the ORIGINAL sparse space, the block
// centroid mu = mean(docs) and the residual radius R = max_d ||d - mu||, plus
// outlier indicators. Then for a sample of queries it compares the bound
//     U = query.mu + ||query|| * R
// against the true block max score  T = max_d (query.d)  -- the looseness U/T is
// exactly the pruning headroom the bound would give.
//
// Block structure (clustering) is identical across summary metrics, so any index
// built with the same block-size works.

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
#include <cmath>
#include <algorithm>
#include <unordered_map>

using namespace seismic;
using TIndex = MyInvertedIndex<float>;
using TDataset = SparseDatasetMut<float>;

struct BlockGeom {
    std::unordered_map<uint32_t, double> mu; // centroid (sparse)
    double mu_norm2 = 0.0;                    // ||mu||^2
    double R = 0.0;                           // max residual norm
    double med_resid = 0.0;                   // median residual norm (outlier check)
    size_t n = 0;
    std::vector<std::pair<std::vector<uint16_t>, std::vector<float>>> docs; // kept for query pass
};

// Build centroid + residual stats for one block [start,end) of packed postings.
static BlockGeom block_geom(const TDataset& fwd, const std::vector<uint64_t>& packed,
                            size_t start, size_t end, bool keep_docs) {
    BlockGeom g;
    g.n = end - start;
    if (g.n == 0) return g;

    std::vector<std::pair<std::vector<uint16_t>, std::vector<float>>> docs;
    docs.reserve(g.n);
    for (size_t i = start; i < end; ++i) {
        auto [off, len] = MyPostingList::unpack_offset_len(packed[i]);
        docs.push_back(fwd.get_with_offset(off, len));
    }
    // centroid
    for (auto& [comps, vals] : docs)
        for (size_t j = 0; j < comps.size(); ++j) g.mu[comps[j]] += vals[j];
    for (auto& kv : g.mu) kv.second /= double(g.n);
    for (auto& kv : g.mu) g.mu_norm2 += kv.second * kv.second;

    // residual norms: ||d-mu||^2 = ||mu||^2 + sum_{c in d}(d[c]^2 - 2 d[c] mu[c])
    std::vector<double> resid;
    resid.reserve(g.n);
    for (auto& [comps, vals] : docs) {
        double r2 = g.mu_norm2;
        for (size_t j = 0; j < comps.size(); ++j) {
            double dv = vals[j], m = g.mu.count(comps[j]) ? g.mu[comps[j]] : 0.0;
            r2 += dv * dv - 2.0 * dv * m;
        }
        if (r2 < 0) r2 = 0;
        resid.push_back(std::sqrt(r2));
    }
    g.R = *std::max_element(resid.begin(), resid.end());
    std::nth_element(resid.begin(), resid.begin() + resid.size() / 2, resid.end());
    g.med_resid = resid[resid.size() / 2];
    if (keep_docs) g.docs = std::move(docs);
    return g;
}

static void pctl(const std::string& name, std::vector<double> v) {
    if (v.empty()) { std::cout << name << ": (none)\n"; return; }
    std::sort(v.begin(), v.end());
    auto P = [&](double p){ return v[std::min(v.size()-1, size_t(p*v.size())) ]; };
    double mean = 0; for (double x : v) mean += x; mean /= v.size();
    std::cout << name << "  n=" << v.size()
              << "  mean=" << mean
              << "  p10=" << P(.10) << "  p50=" << P(.50) << "  p90=" << P(.90)
              << "  p99=" << P(.99) << "  max=" << v.back() << "\n";
}

int main(int argc, char** argv) {
    std::string index_file, query_file;
    size_t list_stride = 50;   // sample every Nth posting list (Part 1)
    size_t n_queries = 100;    // queries to sample (Part 2)
    size_t query_cut = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--index-file" && i+1<argc) index_file = argv[++i];
        else if (a == "--query-file" && i+1<argc) query_file = argv[++i];
        else if (a == "--list-stride" && i+1<argc) list_stride = std::stoul(argv[++i]);
        else if (a == "--n-queries" && i+1<argc) n_queries = std::stoul(argv[++i]);
    }
    if (index_file.empty()) { std::cerr << "need --index-file\n"; return 1; }

    std::cout << "Loading index " << index_file << " ...\n";
    TIndex index;
    { std::ifstream f(index_file + ".index.seismic", std::ios::binary);
      if (!f) { std::cerr << "cannot open index\n"; return 1; }
      cereal::BinaryInputArchive ar(f); ar(index); }
    const TDataset& fwd = index.dataset();
    const auto& lists = index.posting_lists();
    std::cout << "docs=" << index.len() << " dim=" << index.dim()
              << " posting_lists=" << lists.size() << "\n";

    // ---- Part 1: structural radius distribution ----
    std::cout << "\n=== Part 1: per-block radius stats (sampling every " << list_stride << "th list) ===\n";
    std::vector<double> v_size, v_R, v_R_over_munorm, v_R_over_med;
    size_t blocks_seen = 0;
    for (size_t li = 0; li < lists.size(); li += list_stride) {
        const auto& pl = lists[li];
        const auto& packed = pl.packed_postings();
        const auto& boff = pl.block_offsets();
        if (boff.size() < 2) continue;
        for (size_t b = 0; b + 1 < boff.size(); ++b) {
            if (boff[b+1] <= boff[b]) continue;
            BlockGeom g = block_geom(fwd, packed, boff[b], boff[b+1], false);
            if (g.n < 2) continue;
            double munorm = std::sqrt(g.mu_norm2);
            v_size.push_back(double(g.n));
            v_R.push_back(g.R);
            if (munorm > 1e-9) v_R_over_munorm.push_back(g.R / munorm);
            if (g.med_resid > 1e-9) v_R_over_med.push_back(g.R / g.med_resid);
            ++blocks_seen;
        }
    }
    std::cout << "sampled blocks=" << blocks_seen << "\n";
    pctl("block size           ", v_size);
    pctl("R (max residual norm) ", v_R);
    pctl("R / ||mu||            ", v_R_over_munorm);   // scale-free tightness
    pctl("R / median-residual   ", v_R_over_med);      // >>1 => outlier-dominated

    // ---- Part 2: compare bound variants vs true block-max over queries ----
    if (!query_file.empty()) {
        std::cout << "\n=== Part 2: looseness U/T (1=tight) for several bound forms ===\n";
        std::cout << "  CS        = q.mu + ||q||_2 * R          (current isotropic radius)\n";
        std::cout << "  restrict  = q.mu + ||q||_2 * R|_support (oracle: residual restricted to query support)\n";
        std::cout << "  maxsum    = sum_c q[c]*max_d d[c]       (max-summary bound; tight reference)\n";
        TDataset queries = TDataset::read_bin_file(query_file);
        size_t nq = std::min(n_queries, queries.len());
        std::vector<float> dq(index.dim(), 0.0f);
        std::vector<double> L_cs, L_restrict, L_maxsum;
        size_t violations = 0, blocks = 0;
        for (size_t qi = 0; qi < nq; ++qi) {
            auto [qc, qv] = queries.get(qi);
            std::vector<std::pair<float,uint16_t>> qs;
            for (size_t j = 0; j < qc.size(); ++j) qs.push_back({qv[j], qc[j]});
            std::sort(qs.begin(), qs.end(), [](auto&a, auto&b){return a.first>b.first;});
            double qnorm2 = 0;
            for (size_t j = 0; j < qc.size(); ++j) { dq[qc[j]] = qv[j]; qnorm2 += double(qv[j])*qv[j]; }
            double qnorm = std::sqrt(qnorm2);
            size_t cut = std::min(query_cut, qs.size());
            for (size_t t = 0; t < cut; ++t) {
                uint16_t comp = qs[t].second;
                if (comp >= lists.size()) continue;
                const auto& pl = lists[comp];
                const auto& packed = pl.packed_postings();
                const auto& boff = pl.block_offsets();
                for (size_t b = 0; b + 1 < boff.size(); ++b) {
                    if (boff[b+1] <= boff[b]) continue;
                    BlockGeom g = block_geom(fwd, packed, boff[b], boff[b+1], true);
                    if (g.n < 2) continue;
                    // per-doc maps for fast lookup
                    std::vector<std::unordered_map<uint16_t,float>> dm(g.docs.size());
                    for (size_t i = 0; i < g.docs.size(); ++i)
                        for (size_t j = 0; j < g.docs[i].first.size(); ++j)
                            dm[i][g.docs[i].first[j]] = g.docs[i].second[j];

                    double qmu = 0; for (auto& kv : g.mu) qmu += dq[kv.first] * kv.second;

                    // true block max  T = max_d q.d
                    double T = 0;
                    for (auto& [comps, vals] : g.docs) {
                        double s = 0; for (size_t j = 0; j < comps.size(); ++j) s += dq[comps[j]] * vals[j];
                        T = std::max(T, s);
                    }
                    if (T <= 1e-6) { ++blocks; continue; }

                    // CS bound
                    double U_cs = qmu + qnorm * g.R;
                    // sparsity-restricted radius: max_d || (d-mu) restricted to query support ||
                    double Rq = 0;
                    for (auto& m : dm) {
                        double rq2 = 0;
                        for (size_t j = 0; j < qc.size(); ++j) {
                            uint16_t c = qc[j];
                            double dv = m.count(c) ? m[c] : 0.0;
                            double mv = g.mu.count(c) ? g.mu[c] : 0.0;
                            rq2 += (dv - mv) * (dv - mv);
                        }
                        Rq = std::max(Rq, std::sqrt(rq2));
                    }
                    double U_restrict = qmu + qnorm * Rq;
                    // max-summary bound: sum_c q[c] * max_d d[c]
                    double U_maxsum = 0;
                    for (size_t j = 0; j < qc.size(); ++j) {
                        uint16_t c = qc[j]; double maxd = 0;
                        for (auto& m : dm) if (m.count(c)) maxd = std::max(maxd, double(m[c]));
                        U_maxsum += double(qv[j]) * maxd;
                    }
                    ++blocks;
                    L_cs.push_back(U_cs / T);
                    L_restrict.push_back(U_restrict / T);
                    L_maxsum.push_back(U_maxsum / T);
                    if (U_restrict + 1e-3 < T || U_maxsum + 1e-3 < T) ++violations;
                }
            }
            for (size_t j = 0; j < qc.size(); ++j) dq[qc[j]] = 0.0f;
        }
        std::cout << "queries=" << nq << " blocks scored=" << blocks
                  << " bound violations=" << violations << " (should be 0)\n";
        pctl("CS       ", L_cs);
        pctl("restrict ", L_restrict);
        pctl("maxsum   ", L_maxsum);
    }
    return 0;
}
