#include "AstroverbCore.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>

namespace astro {

// ------------------------------------------------------------------ tube models
namespace {
// 12AX7 (Koren) + grid current
constexpr double MU = 100.0, EX = 1.4, KG1 = 1060.0, KP = 600.0, KVB = 300.0;
constexpr double GG = 6.06e-4, CG = 11.8, XI = 1.2;
constexpr double CGP = 1.7e-12, CGK = 1.6e-12, CPK = 0.46e-12;
// EL84 (generic fit)
constexpr double EL_MU = 20.344, EL_EX = 1.100, EL_KG1 = 335.87, EL_KP = 123.23, EL_KVB = 47.16, EL_KG2 = 2117.4;
constexpr double EL_GG = 1.0e-3, EL_CG = 12.0, EL_XI = 1.2;
constexpr double EL_CG1P = 0.5e-12, EL_CG1K = 10.3e-12, EL_CPK = 6.0e-12;

constexpr double RS_PICKUP = 7e3;
constexpr double GAIN_K = 5871.0;
constexpr double I_PRE = 7.6e-3;
constexpr double I_IDLE_SUPPLY = 0.059661;      // idle current from the source, calibrates open-circuit voltage
constexpr double R_RET = 62.5e3;                // reverb return source impedance (pot wiper), approx.

inline double audioTaper(double x) { return (std::pow(100.0, x) - 1.0) / 99.0; }


// ---- fast scalar exp/log (rel. error ~1e-14), branch-light, used in the device models ----
inline double fexp(double x) {
    if (x < -700.0) return 0.0;
    if (x > 700.0) x = 700.0;
    const double LOG2E = 1.4426950408889634, LN2HI = 6.93147180369123816490e-01, LN2LO = 1.90821492927058770002e-10;
    double kd = std::floor(x * LOG2E + 0.5);
    double r = (x - kd * LN2HI) - kd * LN2LO;                       // |r| <= 0.347
    double p = 1.0 / 39916800.0;                                    // Taylor to r^11 (error < 2e-15 rel)
    p = p * r + 1.0 / 3628800.0; p = p * r + 1.0 / 362880.0; p = p * r + 1.0 / 40320.0; p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0; p = p * r + 1.0 / 120.0; p = p * r + 1.0 / 24.0; p = p * r + 1.0 / 6.0;
    p = p * r + 0.5; p = p * r + 1.0; p = p * r + 1.0;
    int64_t bits = (int64_t)(kd + 1023.0) << 52;
    double scale; std::memcpy(&scale, &bits, 8);
    return p * scale;
}
inline double flog(double x) {                                      // x > 0, normal numbers
    int64_t bits; std::memcpy(&bits, &x, 8);
    int e = (int)((bits >> 52) & 0x7ff) - 1023;
    bits = (bits & 0x000fffffffffffffLL) | 0x3ff0000000000000LL;
    double m; std::memcpy(&m, &bits, 8);                            // [1,2)
    if (m > 1.4142135623730951) { m *= 0.5; ++e; }
    double f = (m - 1.0) / (m + 1.0), f2 = f * f;
    double s = 1.0 / 17.0;
    s = s * f2 + 1.0 / 15.0; s = s * f2 + 1.0 / 13.0; s = s * f2 + 1.0 / 11.0; s = s * f2 + 1.0 / 9.0;
    s = s * f2 + 1.0 / 7.0; s = s * f2 + 1.0 / 5.0; s = s * f2 + 1.0 / 3.0; s = s * f2 + 1.0;
    return 2.0 * f * s + e * 0.6931471805599453;
}
inline double flog1pexp(double e) {                                 // log(1+e), e >= 0
    if (e < 1e-5) return e * (1.0 - 0.5 * e + e * e * (1.0 / 3.0));
    return flog(1.0 + e);
}
inline double fpow(double x, double y) { return fexp(y * flog(x)); }

// ---- AVX2 batch math (4 devices per vector). Falls back to scalar code when AVX2 is unavailable. ----
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define ASTRO_AVX2 1
#include <immintrin.h>
typedef __m256d vd;
inline vd vset(double x) { return _mm256_set1_pd(x); }
inline vd vfma(vd a, vd b, vd c) { return _mm256_fmadd_pd(a, b, c); }
inline vd vexp4(vd x) {
    x = _mm256_min_pd(_mm256_max_pd(x, vset(-700.0)), vset(700.0));
    const vd magic = vset(6755399441055744.0);
    vd t = vfma(x, vset(1.4426950408889634), magic);
    vd kd = _mm256_sub_pd(t, magic);
    vd r = _mm256_fnmadd_pd(kd, vset(6.93147180369123816490e-01), x);
    r = _mm256_fnmadd_pd(kd, vset(1.90821492927058770002e-10), r);
    vd p = vset(1.0 / 39916800.0);
    p = vfma(p, r, vset(1.0 / 3628800.0)); p = vfma(p, r, vset(1.0 / 362880.0)); p = vfma(p, r, vset(1.0 / 40320.0));
    p = vfma(p, r, vset(1.0 / 5040.0)); p = vfma(p, r, vset(1.0 / 720.0)); p = vfma(p, r, vset(1.0 / 120.0));
    p = vfma(p, r, vset(1.0 / 24.0)); p = vfma(p, r, vset(1.0 / 6.0)); p = vfma(p, r, vset(0.5));
    p = vfma(p, r, vset(1.0)); p = vfma(p, r, vset(1.0));
    __m256i ti = _mm256_castpd_si256(t), mi = _mm256_castpd_si256(magic);
    __m256i ni = _mm256_slli_epi64(_mm256_add_epi64(_mm256_sub_epi64(ti, mi), _mm256_set1_epi64x(1023)), 52);
    return _mm256_mul_pd(p, _mm256_castsi256_pd(ni));
}
inline vd vlog4(vd x) {                                  // x > 0
    x = _mm256_max_pd(x, vset(1e-300));
    __m256i bits = _mm256_castpd_si256(x);
    __m256i eb = _mm256_srli_epi64(bits, 52);
    vd ed = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(eb, _mm256_set1_epi64x(0x4330000000000000LL))), vset(4503599627370496.0));
    vd e = _mm256_sub_pd(ed, vset(1023.0));
    __m256i mb = _mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi64x(0x000fffffffffffffLL)), _mm256_set1_epi64x(0x3ff0000000000000LL));
    vd m = _mm256_castsi256_pd(mb);
    vd big = _mm256_cmp_pd(m, vset(1.4142135623730951), _CMP_GT_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, vset(0.5)), big);
    e = _mm256_add_pd(e, _mm256_and_pd(big, vset(1.0)));
    vd f = _mm256_div_pd(_mm256_sub_pd(m, vset(1.0)), _mm256_add_pd(m, vset(1.0)));
    vd f2 = _mm256_mul_pd(f, f);
    vd s = vset(1.0 / 17.0);
    s = vfma(s, f2, vset(1.0 / 15.0)); s = vfma(s, f2, vset(1.0 / 13.0)); s = vfma(s, f2, vset(1.0 / 11.0));
    s = vfma(s, f2, vset(1.0 / 9.0)); s = vfma(s, f2, vset(1.0 / 7.0)); s = vfma(s, f2, vset(1.0 / 5.0));
    s = vfma(s, f2, vset(1.0 / 3.0)); s = vfma(s, f2, vset(1.0));
    return vfma(_mm256_mul_pd(vset(2.0), f), s, _mm256_mul_pd(e, vset(0.6931471805599453)));
}
// softplus + sigmoid for 4 lanes
inline void vsoftplus4(vd x, vd& sp, vd& sig) {
    vd big = _mm256_cmp_pd(x, vset(30.0), _CMP_GT_OQ);
    vd e = vexp4(_mm256_min_pd(x, vset(30.0)));
    vd lp = vlog4(_mm256_add_pd(vset(1.0), e));
    vd small = _mm256_cmp_pd(e, vset(1e-5), _CMP_LT_OQ);
    vd ser = _mm256_mul_pd(e, vfma(vfma(e, vset(1.0 / 3.0), vset(-0.5)), e, vset(1.0)));
    lp = _mm256_blendv_pd(lp, ser, small);
    sp = _mm256_blendv_pd(lp, x, big);
    vd sg = _mm256_div_pd(e, _mm256_add_pd(vset(1.0), e));
    sig = _mm256_blendv_pd(sg, vset(1.0), big);
}
#endif

inline void softplus(double x, double& sp, double& sig) {
    if (x > 30.0) { sp = x; sig = 1.0; }
    else { double e = fexp(x); sp = flog1pexp(e); sig = e / (1.0 + e); }
}

// ports: [vpk, vgk]; currents [ip, ig]; derivs a=dip/dvpk, c=dip/dvgk, d=dig/dvgk
inline void triode(double vpk, double vgk, double& ip, double& a, double& c, double& ig, double& d) {
    double s2 = KVB + vpk * vpk, s = std::sqrt(s2);
    double x = KP * (1.0 / MU + vgk / s);
    double sp, sig; softplus(x, sp, sig);
    double e1 = vpk / KP * sp;
    if (e1 <= 0.0) { ip = a = c = 0.0; }
    else {
        ip = 2.0 * fpow(e1, EX) / KG1;
        double dxdp = -KP * vgk * vpk / (s2 * s);
        double de1p = sp / KP + vpk / KP * sig * dxdp;
        double de1g = vpk * sig / s;
        double k = EX * ip / e1;
        a = k * de1p; c = k * de1g;
    }
    double y = CG * vgk;
    if (y < -30.0) { ig = d = 0.0; }
    else {
        double sg, sgs; softplus(y, sg, sgs);
        double q = sg / CG;
        ig = GG * fpow(q, XI);
        d = XI * ig * sgs / q;
    }
}

// ports: [vpk, v1k, v2k]; currents [ip, i1, i2]
inline void pentode(double vpk, double v1, double v2in, double* i, double* D /*3x3 row-major*/) {
    bool clamp = v2in < 1.0;
    double v2 = clamp ? 1.0 : v2in;
    double x = EL_KP * (1.0 / EL_MU + v1 / v2);
    double sp, sig; softplus(x, sp, sig);
    double e1 = v2 / EL_KP * sp;
    for (int k = 0; k < 9; ++k) D[k] = 0.0;
    i[0] = i[2] = 0.0;
    if (e1 > 0.0) {
        double f = 2.0 * fpow(e1, EL_EX);
        double de1_1 = sig;
        double de1_2 = clamp ? 0.0 : (sp / EL_KP - sig * v1 / v2);
        double kf = EL_EX * f / e1;
        double df1 = kf * de1_1, df2 = kf * de1_2;
        if (vpk > 0.0) {
            double A = std::atan(vpk / EL_KVB);
            double dA = (1.0 / EL_KVB) / (1.0 + (vpk / EL_KVB) * (vpk / EL_KVB));
            i[0] = f / EL_KG1 * A;
            D[0] = f / EL_KG1 * dA; D[1] = df1 / EL_KG1 * A; D[2] = df2 / EL_KG1 * A;
        }
        i[2] = f / EL_KG2;
        D[7] = df1 / EL_KG2; D[8] = df2 / EL_KG2;      // row 2 (i2): cols [vpk, v1, v2]
    }
    double y = EL_CG * v1;
    i[1] = 0.0;
    if (y > -30.0) {
        double sg, sgs; softplus(y, sg, sgs);
        double q = sg / EL_CG;
        i[1] = EL_GG * fpow(q, EL_XI);
        D[4] = EL_XI * i[1] * sgs / q;                 // row 1 (i1): d/dv1
    }
}

// ------------------------------------------------------------------ small dense helpers
bool invertMatrix(std::vector<double>& A, int n) {          // Gauss-Jordan, in place
    std::vector<double> B((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) B[(size_t)i * n + i] = 1.0;
    for (int c = 0; c < n; ++c) {
        int piv = c; double best = std::fabs(A[(size_t)c * n + c]);
        for (int r = c + 1; r < n; ++r) { double v = std::fabs(A[(size_t)r * n + c]); if (v > best) { best = v; piv = r; } }
        if (best < 1e-300) return false;
        if (piv != c) for (int k = 0; k < n; ++k) { std::swap(A[(size_t)c * n + k], A[(size_t)piv * n + k]); std::swap(B[(size_t)c * n + k], B[(size_t)piv * n + k]); }
        double inv = 1.0 / A[(size_t)c * n + c];
        for (int k = 0; k < n; ++k) { A[(size_t)c * n + k] *= inv; B[(size_t)c * n + k] *= inv; }
        for (int r = 0; r < n; ++r) {
            if (r == c) continue;
            double f = A[(size_t)r * n + c];
            if (f == 0.0) continue;
            for (int k = 0; k < n; ++k) { A[(size_t)r * n + k] -= f * A[(size_t)c * n + k]; B[(size_t)r * n + k] -= f * B[(size_t)c * n + k]; }
        }
    }
    A.swap(B);
    return true;
}

// solve J x = rhs in place (J destroyed), n small
bool solveSmall(double* J, double* rhs, int n) {
    for (int c = 0; c < n; ++c) {
        int piv = c; double best = std::fabs(J[c * n + c]);
        for (int r = c + 1; r < n; ++r) { double v = std::fabs(J[r * n + c]); if (v > best) { best = v; piv = r; } }
        if (best < 1e-300) return false;
        if (piv != c) { for (int k = 0; k < n; ++k) std::swap(J[c * n + k], J[piv * n + k]); std::swap(rhs[c], rhs[piv]); }
        double inv = 1.0 / J[c * n + c];
        for (int r = c + 1; r < n; ++r) {
            double f = J[r * n + c] * inv;
            if (f == 0.0) continue;
            for (int k = c; k < n; ++k) J[r * n + k] -= f * J[c * n + k];
            rhs[r] -= f * rhs[c];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        double s = rhs[r];
        for (int k = r + 1; k < n; ++k) s -= J[r * n + k] * rhs[k];
        rhs[r] = s / J[r * n + r];
    }
    return true;
}

// LU with partial pivoting, row-major in place
bool luFactor(double* a, int* piv, int n) {
    for (int c = 0; c < n; ++c) {
        int p = c; double best = std::fabs(a[c * n + c]);
        for (int r = c + 1; r < n; ++r) { double v = std::fabs(a[r * n + c]); if (v > best) { best = v; p = r; } }
        if (best < 1e-300) return false;
        piv[c] = p;
        if (p != c) for (int k = 0; k < n; ++k) std::swap(a[c * n + k], a[p * n + k]);
        double inv = 1.0 / a[c * n + c];
        for (int r = c + 1; r < n; ++r) {
            double f = a[r * n + c] * inv;
            a[r * n + c] = f;
            if (f != 0.0) for (int k = c + 1; k < n; ++k) a[r * n + k] -= f * a[c * n + k];
        }
    }
    return true;
}
void luSolve(const double* a, const int* piv, double* b, int n) {
    for (int c = 0; c < n; ++c) { int p = piv[c]; if (p != c) std::swap(b[c], b[p]); }
    for (int r = 1; r < n; ++r) { double s = b[r]; const double* ar = a + r * n; for (int k = 0; k < r; ++k) s -= ar[k] * b[k]; b[r] = s; }
    for (int r = n - 1; r >= 0; --r) { double s = b[r]; const double* ar = a + r * n; for (int k = r + 1; k < n; ++k) s -= ar[k] * b[k]; b[r] = s / ar[r]; }
}

// column-major LU (right-looking), partial pivoting; all inner loops are axpy so they vectorise
bool luFactorC(double* __restrict A, int* piv, int n) {
    for (int k = 0; k < n; ++k) {
        int p = k; double best = std::fabs(A[k * n + k]);
        for (int r = k + 1; r < n; ++r) { double v = std::fabs(A[k * n + r]); if (v > best) { best = v; p = r; } }
        if (best < 1e-300) return false;
        piv[k] = p;
        if (p != k) for (int j = 0; j < n; ++j) std::swap(A[j * n + k], A[j * n + p]);
        const double inv = 1.0 / A[k * n + k];
        double* __restrict ck = &A[k * n];
        for (int r = k + 1; r < n; ++r) ck[r] *= inv;
        for (int j = k + 1; j < n; ++j) {
            double* __restrict cj = &A[j * n];
            const double f = cj[k];
            if (f == 0.0) continue;
            for (int r = k + 1; r < n; ++r) cj[r] -= ck[r] * f;
        }
    }
    return true;
}
void luSolveC(const double* __restrict A, const int* piv, double* __restrict b, int n) {
    for (int k = 0; k < n; ++k) { int p = piv[k]; if (p != k) std::swap(b[k], b[p]); }
    for (int k = 0; k < n; ++k) {
        const double bk = b[k];
        if (bk == 0.0) continue;
        const double* __restrict ck = &A[k * n];
        for (int r = k + 1; r < n; ++r) b[r] -= ck[r] * bk;
    }
    for (int k = n - 1; k >= 0; --k) {
        const double* __restrict ck = &A[k * n];
        const double bk = b[k] / ck[k];
        b[k] = bk;
        for (int r = 0; r < k; ++r) b[r] -= ck[r] * bk;
    }
}
}  // namespace

// ------------------------------------------------------------------ construction
Core::Core() {}

int Core::node(const std::string& s) {
    for (size_t i = 0; i < names_.size(); ++i) if (names_[i] == s) return (int)i;
    names_.push_back(s);
    return (int)names_.size() - 1;
}

int Core::term(const std::string& s) {
    if (s == "gnd") return -(F_GND + 1);
    if (s == "B1")  return -(F_B1 + 1);
    if (s == "B2")  return -(F_B2 + 1);
    if (s == "BM")  return -(F_BM + 1);
    if (s == "VOC") return -(F_VOC + 1);
    return node(s);
}

void Core::buildTopology(const Params& p) {
    names_.clear(); R_.clear(); C_.clear(); T_.clear(); PT_.clear(); IND_.clear();
    auto r = [&](const char* a, const char* b, double v, int tag = T_NONE) { R_.push_back({term(a), term(b), v, tag}); };
    auto c = [&](const char* a, const char* b, double v, int tag = T_NONE) { C_.push_back({term(a), term(b), v, tag}); };
    auto tri = [&](const char* pn, const char* g, const char* k, int mult = 1) {
        int tp = term(pn), tg = term(g), tk = term(k);
        T_.push_back({tp, tg, tk, mult});
        for (int m = 0; m < mult; ++m) { C_.push_back({tg, tp, CGP, T_NONE}); C_.push_back({tg, tk, CGK, T_NONE}); C_.push_back({tp, tk, CPK, T_NONE}); }
    };
    auto pent = [&](const char* pn, const char* g1, const char* g2, const char* k) {
        int tp = term(pn), t1 = term(g1), t2 = term(g2), tk = term(k);
        PT_.push_back({tp, t1, t2, tk});
        C_.push_back({t1, tp, EL_CG1P, T_NONE}); C_.push_back({t1, tk, EL_CG1K, T_NONE}); C_.push_back({tp, tk, EL_CPK, T_NONE});
    };

    // ---- preamp ----
    r("in", "gnd", RS_PICKUP); r("in", "gnd", 1e6); r("in", "g1", 69e3);
    tri("p1", "g1", "k1"); r("p1", "B1", 220e3); r("k1", "gnd", 1.8e3); c("k1", "gnd", 1e-6);
    c("p1", "a0", 22e-9); r("a0", "a1", 1e6);
    r("a1", "w1", 0, T_GAIN_TOP); r("w1", "gnd", 0, T_GAIN_BOT);
    tri("p2", "w1", "k2"); r("p2", "B1", 100e3); r("k2", "gnd", 2.2e3);
    c("p2", "b0", 22e-9); r("b0", "b1", 1e6); c("b0", "b1", 470e-12, T_CB);
    r("b1", "g3", 220e3); r("g3", "gnd", 330e3);
    tri("p3", "g3", "k3"); r("p3", "B2", 100e3); r("k3", "gnd", 1.8e3); c("k3", "gnd", 1e-6);
    c("p3", "h0", 22e-9); r("h0", "g4", 680e3); r("g4", "gnd", 15e3);
    tri("p4", "g4", "k4", 2);                                  // V2b + V3a, grids tied (net C)
    r("p4", "B2", 100e3); r("k4", "gnd", 1e3); c("k4", "gnd", 100e-6);
    tri("B2", "p4", "k5"); r("k5", "gnd", 100e3);
    c("k5", "pt", 470e-12); r("pt", "o", 0, T_TRE_A); r("o", "x", 0, T_TRE_B);
    r("k5", "l", 0, T_SLOPE); c("l", "x", 22e-9); c("l", "y", 22e-9);
    r("x", "y", 0, T_BASS); r("y", "gnd", 0, T_MID);
    r("o", "v", 470e3); r("v", "gnd", 150e3);
    r("v", "wv", 0, T_VOL_TOP); r("wv", "gnd", 0, T_VOL_BOT);
    c("wv", "g6", 22e-9); r("g6", "t0", 1e6);

    // ---- phase inverter ----
    tri("pa", "g6", "ktail"); tri("pb", "gb", "ktail");
    r("pa", "B2", 82e3); r("pb", "B2", 100e3); c("pa", "pb", 120e-12);
    r("ktail", "t0", 470.0); r("gb", "t0", 1e6);
    r("t0", "nn", 10e3); r("nn", "gnd", 4.7e3);
    r("nn", "sh", 0, T_FBR); c("nn", "gb", 100e-9);
    c("nn", "q", 100e-9); r("q", "gnd", 0, T_PRES);

    // ---- output stage ----
    c("pa", "gv6", 22e-9); c("pb", "gv7", 22e-9);
    r("gv6", "BM", 220e3); r("gv7", "BM", 220e3);
    r("gv6", "g16", 2.2e3); r("gv7", "g17", 2.2e3);
    pent("op1", "g16", "s6", "gnd"); pent("op2", "g17", "s7", "gnd");
    r("s6", "b348", 470.0); r("s7", "b348", 470.0);
    r("bp", "VOC", 0, T_SAG); c("bp", "gnd", 47e-6);
    r("bp", "b348", 1e3); c("b348", "gnd", 33e-6);

    // output transformer
    c("op1", "gnd", 1e-9); c("op2", "gnd", 1e-9); c("op1", "op2", 2e-9);
    r("op1", "w1a", 70.0); r("w2b", "op2", 70.0); r("shw", "sh", 0.5);
    {
        const double LPP = 20.0, RAA = 8000.0, ZL = 16.0, K_PS = 0.9993, K12 = 0.9998;
        double Lh = LPP / 4.0, n = std::sqrt(RAA / ZL), Ls = Lh * (2.0 / n) * (2.0 / n);
        double m = K_PS * std::sqrt(Lh * Ls);
        Ind ot; ot.br = {{term("w1a"), term("bp")}, {term("bp"), term("w2b")}, {term("shw"), term("gnd")}};
        ot.L = {Lh, K12 * Lh, m, K12 * Lh, Lh, m, m, m, Ls};
        IND_.push_back(ot);
    }
    if (p.speakerLoad) {
        const double RE = 13.0, LE = 1e-3, F0 = 100.0, QMS = 5.0, RM = 45.0;
        r("sh", "s1", RE);
        Ind le; le.br = {{term("s1"), term("s2")}}; le.L = {LE}; IND_.push_back(le);
        double w0 = 2.0 * 3.14159265358979323846 * F0, Lm = RM / (w0 * QMS), Cm = 1.0 / (w0 * w0 * Lm);
        r("s2", "gnd", RM);
        Ind lm; lm.br = {{term("s2"), term("gnd")}}; lm.L = {Lm}; IND_.push_back(lm);
        c("s2", "gnd", Cm);
    } else {
        r("sh", "gnd", 16.0);
    }

    n_ = (int)names_.size();
    nc_ = (int)C_.size();
    iin_ = node("in"); ig4_ = node("g4"); ish_ = node("sh"); ibp_ = node("bp"); ip3_ = node("p3"); ib348_ = node("b348");
    nb_ = 0; for (auto& I : IND_) nb_ += (int)I.br.size();

    // ports
    ports_.clear();
    for (auto& t : T_) { ports_.push_back({t.p, t.k}); ports_.push_back({t.g, t.k}); }
    for (auto& t : PT_) { ports_.push_back({t.p, t.k}); ports_.push_back({t.g1, t.k}); ports_.push_back({t.g2, t.k}); }
    satPort_ = (int)ports_.size();
    ports_.push_back({term("w1a"), term("w2b")});
    np_ = (int)ports_.size();
    ldN_ = (n_ + 3) & ~3; ldP_ = (np_ + 3) & ~3;
    topoBuilt_ = true;
}

void Core::applyParams(const Params& p) {
    P_ = p;
    const double rsrc = 25.0 * std::pow(16.0, p.sag);
    rail_[F_BM] = p.bias;
    rail_[F_VOC] = 360.0 + rsrc * I_IDLE_SUPPLY;
    double ga = (std::pow(GAIN_K, p.gain) - 1.0) / (GAIN_K - 1.0);
    double rox = 250e3 * audioTaper(p.treble), vg = audioTaper(p.volume);
    for (auto& e : R_) {
        double v = e.val;
        switch (e.tag) {
            case T_GAIN_TOP: v = 1e6 * (1.0 - ga); break;
            case T_GAIN_BOT: v = 1e6 * ga; break;
            case T_TRE_A: v = 250e3 - rox; break;
            case T_TRE_B: v = rox; break;
            case T_BASS: v = 1e6 * audioTaper(p.bass); break;
            case T_MID: v = 25e3 * p.mid; break;
            case T_VOL_TOP: v = 1e6 * (1.0 - vg); break;
            case T_VOL_BOT: v = 1e6 * vg; break;
            case T_PRES: v = 25e3 * audioTaper(1.0 - p.presence); break;
            case T_FBR: v = p.fbR; break;
            case T_SLOPE: v = p.slopeR; break;
            case T_SAG: v = rsrc; break;
            default: break;
        }
        e.val = std::max(v, 1.0);
    }
    for (auto& e : C_) if (e.tag == T_CB) e.val = p.brightC;
}

// ------------------------------------------------------------------ matrix assembly
void Core::assemble() {
    const int N = n_;
    auto fixedV = [&](int t) { return rail_[-t - 1]; };
    auto stamp = [&](std::vector<double>& G, std::vector<double>& b, int ta, int tb, double g) {
        if (ta >= 0) {
            G[(size_t)ta * N + ta] += g;
            if (tb >= 0) G[(size_t)ta * N + tb] -= g; else b[ta] += g * fixedV(tb);
        }
        if (tb >= 0) {
            G[(size_t)tb * N + tb] += g;
            if (ta >= 0) G[(size_t)tb * N + ta] -= g; else b[tb] += g * fixedV(ta);
        }
    };
    std::vector<double> G((size_t)N * N, 0.0), b(N, 0.0);
    for (auto& e : R_) stamp(G, b, e.a, e.b, 1.0 / e.val);
    for (int i = 0; i < N; ++i) G[(size_t)i * N + i] += 1e-12;
    std::vector<double> Gres = G, bres = b;

    // capacitors
    gc_.assign(nc_, 0.0); capA_.assign(nc_, 0); capB_.assign(nc_, 0);
    for (int j = 0; j < nc_; ++j) {
        capA_[j] = C_[j].a; capB_[j] = C_[j].b; gc_[j] = 2.0 * C_[j].val / dt_;
        stamp(G, b, C_[j].a, C_[j].b, gc_[j]);
    }
    // inductors: block-diagonal L, Yind = dt/2 * inv(L)
    brAll_.clear();
    for (auto& I : IND_) for (auto& br : I.br) brAll_.push_back(br);
    std::vector<double> Linv((size_t)nb_ * nb_, 0.0);
    {
        int o = 0;
        for (auto& I : IND_) {
            int s = (int)I.br.size();
            std::vector<double> blk(I.L);
            invertMatrix(blk, s);
            for (int a = 0; a < s; ++a) for (int c2 = 0; c2 < s; ++c2) Linv[(size_t)(o + a) * nb_ + o + c2] = blk[(size_t)a * s + c2];
            o += s;
        }
    }
    Yind_.assign((size_t)nb_ * nb_, 0.0);
    for (size_t k = 0; k < Yind_.size(); ++k) Yind_[k] = 0.5 * dt_ * Linv[k];
    auto stampY = [&](std::vector<double>& Gm, std::vector<double>& bm, const std::vector<double>& Y) {
        for (int k = 0; k < nb_; ++k) for (int l = 0; l < nb_; ++l) {
            double y = Y[(size_t)k * nb_ + l];
            if (y == 0.0) continue;
            int rk[2] = {brAll_[k].first, brAll_[k].second}; double sr[2] = {1.0, -1.0};
            int cl[2] = {brAll_[l].first, brAll_[l].second}; double sc[2] = {1.0, -1.0};
            for (int u = 0; u < 2; ++u) {
                if (rk[u] < 0) continue;
                for (int w = 0; w < 2; ++w) {
                    double coef = sr[u] * sc[w] * y;
                    if (cl[w] >= 0) Gm[(size_t)rk[u] * N + cl[w]] += coef; else bm[rk[u]] -= coef * fixedV(cl[w]);
                }
            }
        }
    };
    stampY(G, b, Yind_);
    b[ib348_] -= I_PRE;
    bt_ = b;

    // port bookkeeping
    portConst_.assign(np_, 0.0);
    for (int q = 0; q < np_; ++q) {
        double c0 = 0.0;
        if (ports_[q].ta < 0) c0 += fixedV(ports_[q].ta);
        if (ports_[q].tb < 0) c0 -= fixedV(ports_[q].tb);
        portConst_[q] = c0;
    }
    auto buildHZ = [&](std::vector<double> M, std::vector<double>& H, std::vector<double>& HC, std::vector<double>& Z) {
        invertMatrix(M, N);
        H = M;
        HC.assign((size_t)N * np_, 0.0); Z.assign((size_t)np_ * np_, 0.0);
        for (int q = 0; q < np_; ++q) {
            int ta = ports_[q].ta, tb = ports_[q].tb;
            for (int i = 0; i < N; ++i) {
                double v = 0.0;
                if (ta >= 0) v += H[(size_t)i * N + ta];
                if (tb >= 0) v -= H[(size_t)i * N + tb];
                HC[(size_t)i * np_ + q] = v;
            }
        }
        for (int p2 = 0; p2 < np_; ++p2) for (int q = 0; q < np_; ++q) {
            double z = 0.0;
            if (ports_[p2].ta >= 0) z += HC[(size_t)ports_[p2].ta * np_ + q];
            if (ports_[p2].tb >= 0) z -= HC[(size_t)ports_[p2].tb * np_ + q];
            Z[(size_t)p2 * np_ + q] = z;
        }
    };
    buildHZ(G, H_, HC_, Z_);
    {   // column-major copies for the per-sample kernels
        std::vector<double> Hc((size_t)ldN_ * ldN_, 0.0), HCc((size_t)ldN_ * ldP_, 0.0), Zc((size_t)ldP_ * ldP_, 0.0);
        for (int r = 0; r < N; ++r) for (int c2 = 0; c2 < N; ++c2) Hc[(size_t)c2 * ldN_ + r] = H_[(size_t)r * N + c2];
        for (int r = 0; r < N; ++r) for (int q = 0; q < np_; ++q) HCc[(size_t)q * ldN_ + r] = HC_[(size_t)r * np_ + q];
        for (int p2 = 0; p2 < np_; ++p2) for (int q = 0; q < np_; ++q) Zc[(size_t)q * ldP_ + p2] = Z_[(size_t)p2 * np_ + q];
        H_.swap(Hc); HC_.swap(HCc); Zc_.swap(Zc);
    }
    factorAge_ = 1000;

    // DC system: inductors are shorts
    std::vector<double> Gd = Gres, bd = bres;
    for (auto& br : brAll_) stamp(Gd, bd, br.first, br.second, 1e5);
    bd[ib348_] -= I_PRE;
    bdc_ = bd;
    buildHZ(Gd, Hdc_, HCdc_, Zdc_);
}

// ------------------------------------------------------------------ nonlinear devices at the ports
void Core::evalDevices(const std::vector<double>& v, std::vector<double>& i, std::vector<double>& D, bool useSat) const {
    ++const_cast<Core*>(this)->statEval;
    // D is np x np dense (only diagonal blocks used)
    std::fill(D.begin(), D.end(), 0.0);
    int q = 0;
    for (auto& t : T_) {
        double ip, a, c, ig, d;
        triode(v[q], v[q + 1], ip, a, c, ig, d);
        double m = (double)t.mult;
        i[q] = m * ip; i[q + 1] = m * ig;
        D[(size_t)q * np_ + q] = m * a; D[(size_t)q * np_ + q + 1] = m * c;
        D[(size_t)(q + 1) * np_ + q + 1] = m * d;
        q += 2;
    }
    for (size_t k = 0; k < PT_.size(); ++k) {
        double cur[3], Dm[9];
        pentode(v[q], v[q + 1], v[q + 2], cur, Dm);
        for (int a = 0; a < 3; ++a) { i[q + a] = cur[a]; for (int b = 0; b < 3; ++b) D[(size_t)(q + a) * np_ + q + b] = Dm[a * 3 + b]; }
        q += 3;
    }
    // saturation
    if (useSat && P_.otSat && satPort_ >= 0) {
        const double Lp = 20.0, lk = 1.4, h = 0.5 * dt_;
        double lam = lam0_ + h * v[satPort_];
        double lc = std::max(-4.0 * lk, std::min(4.0 * lk, lam));
        double r4 = lc / lk; r4 = r4 * r4; r4 = r4 * r4;
        i[satPort_] = (lc / Lp) * r4;
        D[(size_t)satPort_ * np_ + satPort_] = h * 5.0 * r4 / Lp;
    } else if (satPort_ >= 0) {
        i[satPort_] = 0.0;
    }
}

void Core::newtonPorts(const std::vector<double>& v0, std::vector<double>& v, std::vector<double>& i,
                       const std::vector<double>& Z, bool useSat, bool dcMode, int maxit, double tol, bool ls, int& its, bool& ok) {
    const int np = np_;
    ok = false; its = maxit;
    std::vector<double>& D = D_; std::vector<double>& J = J_; std::vector<double>& rr = r_; std::vector<double>& dv = dv_;
    auto residual = [&](const std::vector<double>& vv, std::vector<double>& ii, std::vector<double>& out) {
        for (int p = 0; p < np; ++p) {
            double s = vv[p] - v0[p];
            const double* zr = &Z[(size_t)p * np];
            for (int q = 0; q < np; ++q) s += zr[q] * ii[q];
            out[p] = s;
        }
    };
    for (int it = 0; it < maxit; ++it) {
        evalDevices(v, i, D, useSat);
        residual(v, i, rr);
        // J = I + Z D (block columns)
        for (int p = 0; p < np; ++p) for (int q = 0; q < np; ++q) J[(size_t)p * np + q] = (p == q) ? 1.0 : 0.0;
        int q0 = 0;
        auto addBlock = [&](int q0b, int m) {
            for (int p = 0; p < np; ++p) {
                const double* zr = &Z[(size_t)p * np];
                for (int kc = 0; kc < m; ++kc) {
                    double s = 0.0;
                    for (int l = 0; l < m; ++l) s += zr[q0b + l] * D[(size_t)(q0b + l) * np + q0b + kc];
                    J[(size_t)p * np + q0b + kc] += s;
                }
            }
        };
        for (size_t k = 0; k < T_.size(); ++k) { addBlock(q0, 2); q0 += 2; }
        for (size_t k = 0; k < PT_.size(); ++k) { addBlock(q0, 3); q0 += 3; }
        addBlock(satPort_, 1);
        for (int p = 0; p < np; ++p) dv[p] = -rr[p];
        if (!solveSmall(J.data(), dv.data(), np)) return;
        double mx = 0.0; for (int p = 0; p < np; ++p) mx = std::max(mx, std::fabs(dv[p]));
        if (mx > 25.0) { double s = 25.0 / mx; for (int p = 0; p < np; ++p) dv[p] *= s; }
        double a = 1.0;
        if (ls) {
            double n0 = 0.0; for (int p = 0; p < np; ++p) n0 += rr[p] * rr[p];
            std::vector<double> vt(np), it2(np), rt(np);
            for (int tries = 0; tries < 16; ++tries) {
                for (int p = 0; p < np; ++p) vt[p] = v[p] + a * dv[p];
                std::vector<double> Dt(D);
                evalDevices(vt, it2, Dt, useSat);
                residual(vt, it2, rt);
                double n1 = 0.0; for (int p = 0; p < np; ++p) n1 += rt[p] * rt[p];
                if (n1 < n0 || n0 < 1e-26) break;
                a *= 0.5;
            }
        }
        double step = 0.0;
        for (int p = 0; p < np; ++p) { v[p] += a * dv[p]; step = std::max(step, std::fabs(a * dv[p])); }
        if (step < tol) { ok = true; its = it + 1; return; }
    }
    (void)dcMode;
}


void Core::setTol() {
    invTol_.assign(ldP_, 0.0);
    int q = 0;
    // triodes in build order: V1a, V1b, V2a, V2b|V3a, V3b (small-signal preamp) then the two phase-inverter halves
    for (size_t k = 0; k < T_.size(); ++k, q += 2) {
        double t = (k < 5) ? 1e-6 : 1e-5;
        invTol_[q] = invTol_[q + 1] = 1.0 / (t * tolScale_);
    }
    for (size_t k = 0; k < PT_.size(); ++k, q += 3) {
        invTol_[q] = 1.0 / (1e-4 * tolScale_); invTol_[q + 1] = 1.0 / (1e-5 * tolScale_); invTol_[q + 2] = 1.0 / (1e-4 * tolScale_);
    }
    invTol_[q] = 1.0 / (1e-4 * tolScale_);    // saturation port
}

// ------------------------------------------------------------------ chord Newton for the transient
void Core::evalFast(const double* v, double* iq) {
    ++statEval;
    const size_t nT = T_.size(), nP = PT_.size();
#ifdef ASTRO_AVX2
    {
        double vpk[8], vgk[8], mu_[8], ip[8], ig[8], a[8], cc[8], d[8];
        for (int k = 0; k < 8; ++k) { vpk[k] = 100.0; vgk[k] = -1.0; mu_[k] = 0.0; }
        for (size_t k = 0; k < nT; ++k) { vpk[k] = v[2 * k]; vgk[k] = v[2 * k + 1]; mu_[k] = (double)T_[k].mult; }
        for (int g = 0; g < 8; g += 4) {
            if ((size_t)g >= nT) break;
            vd Vp = _mm256_loadu_pd(vpk + g), Vg = _mm256_loadu_pd(vgk + g);
            vd s2 = vfma(Vp, Vp, vset(KVB)), s = _mm256_sqrt_pd(s2);
            vd x = vfma(vset(KP), _mm256_div_pd(Vg, s), vset(KP / MU));
            vd sp, sig; vsoftplus4(x, sp, sig);
            vd e1 = _mm256_mul_pd(_mm256_mul_pd(Vp, sp), vset(1.0 / KP));
            vd pos = _mm256_cmp_pd(e1, vset(0.0), _CMP_GT_OQ);
            vd IP = _mm256_and_pd(pos, _mm256_mul_pd(vset(2.0 / KG1), vexp4(_mm256_mul_pd(vset(EX), vlog4(e1)))));
            vd dxdp = _mm256_div_pd(_mm256_mul_pd(vset(-KP), _mm256_mul_pd(Vg, Vp)), _mm256_mul_pd(s2, s));
            vd de1p = vfma(_mm256_mul_pd(_mm256_mul_pd(Vp, sig), vset(1.0 / KP)), dxdp, _mm256_mul_pd(sp, vset(1.0 / KP)));
            vd de1g = _mm256_div_pd(_mm256_mul_pd(Vp, sig), s);
            vd kk = _mm256_div_pd(_mm256_mul_pd(vset(EX), IP), _mm256_max_pd(e1, vset(1e-300)));
            vd A = _mm256_mul_pd(kk, de1p), Cc = _mm256_mul_pd(kk, de1g);
            vd y = _mm256_max_pd(_mm256_mul_pd(vset(CG), Vg), vset(-30.0));
            vd sg, sgs; vsoftplus4(y, sg, sgs);
            vd q = _mm256_mul_pd(sg, vset(1.0 / CG));
            vd IG = _mm256_mul_pd(vset(GG), vexp4(_mm256_mul_pd(vset(XI), vlog4(q))));
            vd Dd = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(vset(XI), IG), sgs), q);
            _mm256_storeu_pd(ip + g, IP); _mm256_storeu_pd(ig + g, IG);
            _mm256_storeu_pd(a + g, A); _mm256_storeu_pd(cc + g, Cc); _mm256_storeu_pd(d + g, Dd);
        }
        for (size_t k = 0; k < nT; ++k) {
            double m = mu_[k];
            iq[2 * k] = m * ip[k]; iq[2 * k + 1] = m * ig[k];
            dT_[3 * k] = m * a[k]; dT_[3 * k + 1] = m * cc[k]; dT_[3 * k + 2] = m * d[k];
        }
    }
    if (nP > 0) {
        double vp[4] = {100, 100, 100, 100}, v1[4] = {-1, -1, -1, -1}, v2[4] = {300, 300, 300, 300};
        const size_t q0 = 2 * nT;
        for (size_t k = 0; k < nP && k < 4; ++k) { vp[k] = v[q0 + 3 * k]; v1[k] = v[q0 + 3 * k + 1]; v2[k] = v[q0 + 3 * k + 2]; }
        vd Vp = _mm256_loadu_pd(vp), V1 = _mm256_loadu_pd(v1), V2i = _mm256_loadu_pd(v2);
        vd clampM = _mm256_cmp_pd(V2i, vset(1.0), _CMP_LT_OQ);
        vd V2 = _mm256_max_pd(V2i, vset(1.0));
        vd x = vfma(vset(EL_KP), _mm256_div_pd(V1, V2), vset(EL_KP / EL_MU));
        vd sp, sig; vsoftplus4(x, sp, sig);
        vd e1 = _mm256_mul_pd(_mm256_mul_pd(V2, sp), vset(1.0 / EL_KP));
        vd f = _mm256_mul_pd(vset(2.0), vexp4(_mm256_mul_pd(vset(EL_EX), vlog4(e1))));
        vd de1_2 = _mm256_andnot_pd(clampM, _mm256_sub_pd(_mm256_mul_pd(sp, vset(1.0 / EL_KP)), _mm256_div_pd(_mm256_mul_pd(sig, V1), V2)));
        vd kf = _mm256_div_pd(_mm256_mul_pd(vset(EL_EX), f), _mm256_max_pd(e1, vset(1e-300)));
        vd df1 = _mm256_mul_pd(kf, sig), df2 = _mm256_mul_pd(kf, de1_2);
        vd y = _mm256_max_pd(_mm256_mul_pd(vset(EL_CG), V1), vset(-30.0));
        vd sg, sgs; vsoftplus4(y, sg, sgs);
        vd qq = _mm256_mul_pd(sg, vset(1.0 / EL_CG));
        vd I1 = _mm256_mul_pd(vset(EL_GG), vexp4(_mm256_mul_pd(vset(EL_XI), vlog4(qq))));
        vd D1 = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(vset(EL_XI), I1), sgs), qq);
        double F[4], DF1[4], DF2[4], I1a[4], D1a[4];
        _mm256_storeu_pd(F, f); _mm256_storeu_pd(DF1, df1); _mm256_storeu_pd(DF2, df2); _mm256_storeu_pd(I1a, I1); _mm256_storeu_pd(D1a, D1);
        for (size_t k = 0; k < nP && k < 4; ++k) {
            double* Dm = &dP_[9 * k];
            for (int z = 0; z < 9; ++z) Dm[z] = 0.0;
            double ipk = 0.0;
            if (vp[k] > 0.0) {
                double A = std::atan(vp[k] / EL_KVB);
                double dA = (1.0 / EL_KVB) / (1.0 + (vp[k] / EL_KVB) * (vp[k] / EL_KVB));
                ipk = F[k] / EL_KG1 * A;
                Dm[0] = F[k] / EL_KG1 * dA; Dm[1] = DF1[k] / EL_KG1 * A; Dm[2] = DF2[k] / EL_KG1 * A;
            }
            Dm[4] = D1a[k]; Dm[7] = DF1[k] / EL_KG2; Dm[8] = DF2[k] / EL_KG2;
            iq[q0 + 3 * k] = ipk; iq[q0 + 3 * k + 1] = I1a[k]; iq[q0 + 3 * k + 2] = F[k] / EL_KG2;
        }
    }
#else
    {
        for (size_t k = 0; k < nT; ++k) {
            double ip, a, c2, ig, d; triode(v[2 * k], v[2 * k + 1], ip, a, c2, ig, d);
            double m = (double)T_[k].mult;
            iq[2 * k] = m * ip; iq[2 * k + 1] = m * ig; dT_[3 * k] = m * a; dT_[3 * k + 1] = m * c2; dT_[3 * k + 2] = m * d;
        }
        const size_t q0 = 2 * nT;
        for (size_t k = 0; k < nP; ++k) pentode(v[q0 + 3 * k], v[q0 + 3 * k + 1], v[q0 + 3 * k + 2], &iq[q0 + 3 * k], &dP_[9 * k]);
    }
#endif
    // output-transformer saturation
    if (P_.otSat) {
        const double Lp = 20.0, lk = 1.4, h = 0.5 * dt_;
        double lam = lam0_ + h * v[satPort_];
        double lc = std::max(-4.0 * lk, std::min(4.0 * lk, lam));
        double r4 = lc / lk; r4 = r4 * r4; r4 = r4 * r4;
        iq[satPort_] = (lc / Lp) * r4; dS_ = h * 5.0 * r4 / Lp;
    } else { iq[satPort_] = 0.0; dS_ = 0.0; }
}

void Core::refactor() {
    ++statRefactor;
    const int np = ldP_;
    LU_.assign((size_t)np * np, 0.0);
    double* J = LU_.data();
    for (int p = 0; p < np; ++p) J[(size_t)p * np + p] = 1.0;
    const double* Zc = Zc_.data();
    int q = 0;
    for (size_t k = 0; k < T_.size(); ++k, q += 2) {
        const double a = dT_[3 * k], c2 = dT_[3 * k + 1], d = dT_[3 * k + 2];
        const double* z0 = Zc + (size_t)q * np; const double* z1 = z0 + np;
        double* j0 = J + (size_t)q * np; double* j1 = j0 + np;
        for (int p = 0; p < np; ++p) { j0[p] += z0[p] * a; j1[p] += z0[p] * c2 + z1[p] * d; }
    }
    for (size_t k = 0; k < PT_.size(); ++k, q += 3) {
        const double* Dm = &dP_[9 * k];
        for (int kc = 0; kc < 3; ++kc) {
            double* jc = J + (size_t)(q + kc) * np;
            for (int l = 0; l < 3; ++l) {
                const double coef = Dm[l * 3 + kc];
                if (coef == 0.0) continue;
                const double* zl = Zc + (size_t)(q + l) * np;
                for (int p = 0; p < np; ++p) jc[p] += zl[p] * coef;
            }
        }
    }
    {
        const double* zs = Zc + (size_t)satPort_ * np; double* js = J + (size_t)satPort_ * np;
        for (int p = 0; p < np; ++p) js[p] += zs[p] * dS_;
    }
    luFactorC(J, piv_.data(), np);
}

void Core::newtonChord(const std::vector<double>& v0, int& its, bool& ok) {
    const int np = ldP_, npr = np_;
    double* __restrict v = v_.data(); double* __restrict dv = dv_.data(); double* __restrict rr = r_.data();
    double* iq = iPort_.data();
    if (kExtrapolate) for (int q = 0; q < npr; ++q) v[q] = 2.0 * vPrev1_[q] - vPrev2_[q];
    ok = false; its = kMaxIt; double prevMx = 1e30;
    evalFast(v, iq);
    if (factorAge_ >= kRefactorEvery) { refactor(); factorAge_ = 0; }
    for (int it = 0; it < kMaxIt; ++it) {
        for (int p = 0; p < np; ++p) rr[p] = v[p] - v0[p];
        for (int q = 0; q < npr; ++q) {
            const double ik = iq[q];
            if (ik == 0.0) continue;
            const double* __restrict zc = &Zc_[(size_t)q * np];
            for (int p = 0; p < np; ++p) rr[p] += zc[p] * ik;
        }
        for (int p = 0; p < np; ++p) dv[p] = -rr[p];
        luSolveC(LU_.data(), piv_.data(), dv, np);
        double mx = 0.0, mxw = 0.0;
        for (int p = 0; p < np; ++p) { double a = std::fabs(dv[p]); mx = std::max(mx, a); mxw = std::max(mxw, a * invTol_[p]); }
        if (mx > 25.0) { double s = 25.0 / mx; for (int p = 0; p < np; ++p) dv[p] *= s; mx = 25.0; mxw *= s; }
        for (int p = 0; p < np; ++p) v[p] += dv[p];
        if (mxw < 1.0) { ok = true; its = it + 1; break; }
        evalFast(v, iq);
        if (it >= 1 ? (mxw > kContract * prevMx) : (mxw > kFirstStep)) { refactor(); factorAge_ = 0; }
        prevMx = mxw;
    }
    ++factorAge_;
    if (its >= 4) factorAge_ = kRefactorEvery;
}

// ------------------------------------------------------------------ DC operating point
void Core::dcSolve() {
    const int N = n_, np = np_;
    std::vector<double> y(N, 0.0), v0(np), v(np, 0.0), i(np, 0.0);
    for (int r = 0; r < N; ++r) { double s = 0.0; for (int c2 = 0; c2 < N; ++c2) s += Hdc_[(size_t)r * N + c2] * bdc_[c2]; y[r] = s; }
    for (int q = 0; q < np; ++q) {
        double s = portConst_[q];
        if (ports_[q].ta >= 0) s += y[ports_[q].ta];
        if (ports_[q].tb >= 0) s -= y[ports_[q].tb];
        v0[q] = s;
    }
    int its; bool ok;
    D_.assign((size_t)np * np, 0.0); J_.assign((size_t)np * np, 0.0); dv_.assign(np, 0.0); r_.assign(np, 0.0);
    newtonPorts(v0, v, i, Zdc_, false, true, 400, 1e-9, true, its, ok);
    dcOk_ = ok;
    x_.assign(ldN_, 0.0);
    for (int r = 0; r < N; ++r) { double s = y[r]; for (int q = 0; q < np; ++q) s -= HCdc_[(size_t)r * np + q] * i[q]; x_[r] = s; }
    v_.assign(ldP_, 0.0); iPort_.assign(ldP_, 0.0);
    for (int q = 0; q < np; ++q) { v_[q] = v[q]; iPort_[q] = i[q]; }
}

void Core::resetStates() {
    dcSolve();
    auto V = [&](int t) { return t >= 0 ? x_[t] : rail_[-t - 1]; };
    cv_.assign(nc_, 0.0); ci_.assign(nc_, 0.0);
    for (int j = 0; j < nc_; ++j) cv_[j] = V(capA_[j]) - V(capB_[j]);
    bv_.assign(nb_, 0.0); bi_.assign(nb_, 0.0);
    for (int k = 0; k < nb_; ++k) bv_[k] = V(brAll_[k].first) - V(brAll_[k].second);
    // initial inductor currents: OT windings carry the DC plate currents
    auto Vn = [&](const char* nm) { return x_[node(nm)]; };
    bi_[0] = (Vn("op1") - Vn("w1a")) / 70.0;
    bi_[1] = (Vn("w2b") - Vn("op2")) / 70.0;
    bi_[2] = (Vn("sh") - Vn("shw")) / 0.5;
    lam_ = 0.0; vsat_ = V(ports_[satPort_].ta) - V(ports_[satPort_].tb);
    lam0_ = 0.0;
    for (int q = 0; q < np_; ++q) { v_[q] = portConst_[q] + (ports_[q].ta >= 0 ? x_[ports_[q].ta] : 0.0) - (ports_[q].tb >= 0 ? x_[ports_[q].tb] : 0.0); vPrev1_[q] = vPrev2_[q] = v_[q]; }
    sendV_ = x_[ip3_]; bpV_ = x_[ibp_];
}

void Core::prepare(double solverRate, const Params& p) {
    fs_ = solverRate; dt_ = 1.0 / solverRate;
    buildTopology(p);
    b_.assign(ldN_, 0.0); y_.assign(ldN_, 0.0); v0_.assign(ldP_, 0.0);
    D_.assign((size_t)np_ * np_, 0.0); J_.assign((size_t)np_ * np_, 0.0); dv_.assign(ldP_, 0.0); r_.assign(ldP_, 0.0);
    iPort_.assign(ldP_, 0.0); dT_.assign(3 * T_.size() + 3, 0.0); dP_.assign(9 * PT_.size() + 9, 0.0); piv_.assign(ldP_, 0);
    applyParams(p);
    assemble();
    setTol();
    resetStates();
    fails_ = 0;
}

void Core::setParams(const Params& p) {
    bool needReset = (p.speakerLoad != P_.speakerLoad);
    if (needReset) { prepare(fs_, p); return; }
    applyParams(p);
    assemble();
}

double Core::nodeV(const std::string& name) const {
    for (size_t i = 0; i < names_.size(); ++i) if (names_[i] == name) return x_[i];
    return 0.0;
}

// ------------------------------------------------------------------ per-sample processing
double Core::process(double vin, double vret) {
    const int N = n_, np = np_, LN = ldN_;
    auto V = [&](int t) { return t >= 0 ? x_[t] : rail_[-t - 1]; };
    double* b = b_.data();
    std::memcpy(b, bt_.data(), sizeof(double) * LN);
    // capacitor histories
    for (int j = 0; j < nc_; ++j) {
        double ieq = gc_[j] * cv_[j] + ci_[j];
        if (capA_[j] >= 0) b[capA_[j]] += ieq;
        if (capB_[j] >= 0) b[capB_[j]] -= ieq;
    }
    // inductor histories
    for (int k = 0; k < nb_; ++k) {
        double ih = bi_[k];
        const double* yr = &Yind_[(size_t)k * nb_];
        for (int l = 0; l < nb_; ++l) ih += yr[l] * bv_[l];
        if (brAll_[k].first >= 0) b[brAll_[k].first] -= ih;
        if (brAll_[k].second >= 0) b[brAll_[k].second] += ih;
    }
    b[iin_] += vin / RS_PICKUP;
    if (vret != 0.0 && P_.revLevel > 0.0) b[ig4_] += audioTaper(P_.revLevel) * vret / R_RET;

    // linear prediction y = H b (column-major axpy form, vectorises) ; port voltages
    double* y = y_.data();
    for (int r = 0; r < LN; ++r) y[r] = 0.0;
    {
        double* __restrict yy = y;
        for (int c2 = 0; c2 < N; ++c2) {
            const double bc = b[c2];
            const double* __restrict hc = &H_[(size_t)c2 * LN];
            for (int r = 0; r < LN; ++r) yy[r] += hc[r] * bc;
        }
    }
    for (int q = 0; q < np; ++q) {
        double s = portConst_[q];
        if (ports_[q].ta >= 0) s += y[ports_[q].ta];
        if (ports_[q].tb >= 0) s -= y[ports_[q].tb];
        v0_[q] = s;
    }
    lam0_ = lam_ + 0.5 * dt_ * vsat_;
    int its; bool ok;
    if (kFullNewton) newtonPorts(v0_, v_, iPort_, Z_, true, false, 30, 1e-6, false, its, ok);
    else newtonChord(v0_, its, ok);
    if (!ok && !kFullNewton)
    {   // rare hard sample: restart from the previous solution with damped full Newton (line search) for robustness
        for (int q = 0; q < np; ++q) v_[q] = vPrev1_[q];
        ++statFallback;
        newtonPorts(v0_, v_, iPort_, Z_, true, false, 120, 1e-6, true, its, ok);
        factorAge_ = kRefactorEvery;
    }
    lastIts_ = its; if (!ok) ++fails_;
    for (int r = 0; r < LN; ++r) x_[r] = y[r];
    for (int q = 0; q < np; ++q) {
        const double iq = iPort_[q];
        if (iq == 0.0) continue;
        const double* __restrict hq = &HC_[(size_t)q * LN];
        double* __restrict xr = x_.data();
        for (int r = 0; r < LN; ++r) xr[r] -= hq[r] * iq;
    }
    double vnew = v_[satPort_];
    lam_ = lam0_ + 0.5 * dt_ * vnew; vsat_ = vnew;
    for (int q = 0; q < np; ++q) { vPrev2_[q] = vPrev1_[q]; vPrev1_[q] = v_[q]; }
    for (int j = 0; j < nc_; ++j) {
        double ieq = gc_[j] * cv_[j] + ci_[j];
        double cvn = V(capA_[j]) - V(capB_[j]);
        ci_[j] = gc_[j] * cvn - ieq; cv_[j] = cvn;
    }
    {
        double bvn[16];
        for (int k = 0; k < nb_; ++k) bvn[k] = V(brAll_[k].first) - V(brAll_[k].second);
        double inc[16];
        for (int k = 0; k < nb_; ++k) { double s = 0.0; for (int l = 0; l < nb_; ++l) s += Yind_[(size_t)k * nb_ + l] * (bv_[l] + bvn[l]); inc[k] = s; }
        for (int k = 0; k < nb_; ++k) { bi_[k] += inc[k]; bv_[k] = bvn[k]; }
    }
    sendV_ = x_[ip3_]; bpV_ = x_[ibp_];
    double out = x_[ish_];
    if (!(out == out) || std::fabs(out) > 1e5) { resetStates(); out = 0.0; }
    return out;
}

}  // namespace astro
