#include "../src/flashmp/_csrc/flashmp.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

static double brute_cov(const double* x, int64_t i, const double* y, int64_t j, int64_t m) {
    double mx = 0.0, my = 0.0;
    for (int64_t k = 0; k < m; ++k) { mx += x[i + k]; my += y[j + k]; }
    mx /= static_cast<double>(m);
    my /= static_cast<double>(m);
    double vx = 0.0, vy = 0.0, c = 0.0;
    for (int64_t k = 0; k < m; ++k) {
        double dx = x[i + k] - mx;
        double dy = y[j + k] - my;
        vx += dx * dx;
        vy += dy * dy;
        c += dx * dy;
    }
    if (vx <= 1e-24 || vy <= 1e-24) return std::numeric_limits<double>::quiet_NaN();
    return c / std::sqrt(vx * vy);
}

int main() {
    std::mt19937_64 rng(1337);
    std::normal_distribution<double> nd(0.0, 1.0);

    // 1. Self-join vs brute force on small N with edge lengths
    for (int64_t N : {37, 128, 311}) {
        int64_t m = 11;
        int64_t n = N - m + 1;
        int64_t ez = fmp_default_excl_zone(m);
        std::vector<double> T(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i) T[static_cast<size_t>(i)] = nd(rng);
        std::vector<double> mp(static_cast<size_t>(n)), lmp(static_cast<size_t>(n)), rmp(static_cast<size_t>(n));
        std::vector<int64_t> ix(static_cast<size_t>(n)), lix(static_cast<size_t>(n)), rix(static_cast<size_t>(n));
        int rc = fmp_self_join(T.data(), N, m, -1, 2, mp.data(), ix.data(), lmp.data(), lix.data(), rmp.data(), rix.data());
        assert(rc == 0);
        for (int64_t i = 0; i < n; ++i) {
            double best_r = -2.0;
            for (int64_t j = 0; j < n; ++j) {
                if (std::abs(i - j) <= ez) continue;
                double r = brute_cov(T.data(), i, T.data(), j, m);
                if (r > best_r) best_r = r;
            }
            double exp_d = std::sqrt(std::max(0.0, 2.0 * static_cast<double>(m) * (1.0 - std::clamp(best_r, -1.0, 1.0))));
            assert(std::abs(mp[static_cast<size_t>(i)] - exp_d) < 1e-11);
        }
    }

    // 2. NaN, Inf, flat region, tiny-scale (1e-10), and high-DC (1e10) edge cases
    {
        int64_t N = 200, m = 16, n = N - m + 1;
        std::vector<double> T(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i) T[static_cast<size_t>(i)] = std::sin(0.2 * static_cast<double>(i)) + 0.05 * nd(rng);
        T[50] = std::numeric_limits<double>::quiet_NaN();
        T[120] = std::numeric_limits<double>::infinity();
        for (int64_t i = 80; i < 105; ++i) T[static_cast<size_t>(i)] = 42.0; // flat region
        std::vector<double> mp(static_cast<size_t>(n));
        std::vector<int64_t> ix(static_cast<size_t>(n));
        assert(fmp_self_join(T.data(), N, m, -1, 2, mp.data(), ix.data(), nullptr, nullptr, nullptr, nullptr) == 0);
        // Subsequences overlapping NaN(50) or Inf(120) or flat[80..104] must have idx == -1 and mp == inf
        assert(ix[50] == -1 && std::isinf(mp[50]));
        assert(ix[120] == -1 && std::isinf(mp[120]));
        assert(ix[82] == -1 && std::isinf(mp[82]));
        // Clean subsequences must remain finite
        assert(ix[0] >= 0 && std::isfinite(mp[0]));
    }

    // 3. AB-join & LeakScan (affine & sign-flipped copy)
    {
        int64_t NA = 300, NB = 120, m = 24;
        std::vector<double> A(static_cast<size_t>(NA)), B(static_cast<size_t>(NB));
        for (auto& v : A) v = nd(rng);
        for (auto& v : B) v = nd(rng);
        // Embed an affine-transformed copy of A[70..94] into B[40..64]
        for (int64_t k = 0; k < m; ++k) B[static_cast<size_t>(40 + k)] = -3.7 * A[static_cast<size_t>(70 + k)] + 999.5;
        std::vector<double> bc(static_cast<size_t>(NB - m + 1));
        std::vector<int64_t> bi(static_cast<size_t>(NB - m + 1));
        int64_t hits = 0;
        assert(fmp_leak_scan(A.data(), NA, B.data(), NB, m, 0.9999, 1, 2, bc.data(), bi.data(), &hits) == 0);
        assert(hits >= 1);
        assert(bi[40] == 70);
        assert(bc[40] > 0.999999);
    }

    // 4. mSTUMP on D=5, N=180, m=16
    {
        int64_t D = 5, N = 180, m = 16, n = N - m + 1;
        std::vector<double> M(static_cast<size_t>(D * N));
        for (auto& v : M) v = nd(rng);
        std::vector<double> mp(static_cast<size_t>(D * n));
        std::vector<int64_t> ix(static_cast<size_t>(D * n));
        assert(fmp_mstump(M.data(), D, N, m, -1, 2, mp.data(), ix.data()) == 0);
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t k = 1; k < D; ++k) {
                assert(mp[static_cast<size_t>(k * n + i)] + 1e-12 >= mp[static_cast<size_t>((k - 1) * n + i)]);
            }
        }
    }

    // 5. RingMP exact egress & egress=0 growth + adversarial monotone stack overflow
    {
        int64_t m = 16, W_sub = 90;
        std::vector<double> stream(420);
        for (int64_t i = 0; i < 420; ++i) {
            // Monotonically drifting sinusoid to stress suffix-max stack depth & truncation rebuild
            stream[static_cast<size_t>(i)] = std::sin(0.15 * static_cast<double>(i)) + 0.01 * static_cast<double>(i) + 0.1 * nd(rng);
        }
        fmp_stream_t* s = fmp_stream_create(m, W_sub, -1, 1, 0.05, 1);
        assert(s != nullptr);
        fmp_tick_result_t tr{};
        for (int64_t t = 0; t < 420; ++t) {
            assert(fmp_stream_update(s, stream[static_cast<size_t>(t)], &tr) == 0);
            if (t >= 180 && (t % 25 == 0)) {
                int64_t alen = fmp_stream_active_len(s);
                assert(alen == W_sub);
                std::vector<double> smp(static_cast<size_t>(alen));
                std::vector<int64_t> six(static_cast<size_t>(alen));
                assert(fmp_stream_snapshot(s, 1, smp.data(), six.data(), nullptr, nullptr) == 0);

                int64_t raw_start = t + 1 - (W_sub + m - 1);
                std::vector<double> ref_mp(static_cast<size_t>(alen));
                std::vector<int64_t> ref_ix(static_cast<size_t>(alen));
                assert(fmp_self_join(stream.data() + raw_start, W_sub + m - 1, m, -1, 1,
                                     ref_mp.data(), ref_ix.data(), nullptr, nullptr, nullptr, nullptr) == 0);
                for (int64_t i = 0; i < alen; ++i) {
                    assert(std::abs(smp[static_cast<size_t>(i)] - ref_mp[static_cast<size_t>(i)]) < 1e-11);
                }
            }
        }
        fmp_stream_destroy(s);

        // Unbounded egress=0 growth across power-of-2 capacity boundary
        fmp_stream_t* s0 = fmp_stream_create(12, 16, -1, 0, 0.05, 1);
        assert(s0 != nullptr);
        for (int64_t t = 0; t < 160; ++t) {
            assert(fmp_stream_update(s0, stream[static_cast<size_t>(t)], &tr) == 0);
        }
        int64_t al0 = fmp_stream_active_len(s0);
        assert(al0 == 160 - 12 + 1);
        std::vector<double> smp0(static_cast<size_t>(al0)), ref0(static_cast<size_t>(al0));
        std::vector<int64_t> six0(static_cast<size_t>(al0)), rix0(static_cast<size_t>(al0));
        assert(fmp_stream_snapshot(s0, 1, smp0.data(), six0.data(), nullptr, nullptr) == 0);
        assert(fmp_self_join(stream.data(), 160, 12, -1, 1, ref0.data(), rix0.data(), nullptr, nullptr, nullptr, nullptr) == 0);
        for (int64_t i = 0; i < al0; ++i) {
            assert(std::abs(smp0[static_cast<size_t>(i)] - ref0[static_cast<size_t>(i)]) < 1e-11);
        }
        fmp_stream_destroy(s0);
    }

    // 6. FLOSS & auto_window
    {
        std::vector<double> x(600);
        for (int64_t i = 0; i < 600; ++i) x[static_cast<size_t>(i)] = std::sin(2.0 * 3.141592653589793 * static_cast<double>(i) / 40.0);
        int64_t aw = fmp_auto_window(x.data(), 600, 8, 128);
        assert(aw >= 36 && aw <= 44);
        int64_t n = 600 - aw + 1;
        std::vector<double> mp(static_cast<size_t>(n)), cac(static_cast<size_t>(n));
        std::vector<int64_t> ix(static_cast<size_t>(n));
        assert(fmp_self_join(x.data(), 600, aw, -1, 1, mp.data(), ix.data(), nullptr, nullptr, nullptr, nullptr) == 0);
        assert(fmp_floss_cac(ix.data(), n, aw, cac.data()) == 0);
        for (double v : cac) assert(v >= 0.0 && v <= 1.0);
    }

    std::printf("ALL C++ ASAN/UBSAN HARNESS CHECKS PASSED!\n");
    return 0;
}
