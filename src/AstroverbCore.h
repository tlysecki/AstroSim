// Astroverb 16 circuit core: full-network nonlinear solver (MNA + trapezoidal companions + Newton).
// No JUCE dependency. Port of amp_sim.py; see that file for the circuit and modelling notes.
#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace astro {

struct Params {
    double gain = 0.7, bass = 0.5, mid = 0.5, treble = 0.5, volume = 0.7, presence = 0.5;  // pot rotations 0..1
    double fbR = 10e3;          // global feedback resistor, ohms
    double bias = -16.0;        // EL84 grid bias rail, volts
    double sag = 0.75;          // 0..1 supply source resistance 25 * 16^sag ohms
    double slopeR = 47e3;       // tone-stack slope resistor
    double brightC = 470e-12;   // volume bright cap (farads)
    double revLevel = 0.0;      // reverb return level 0..1 (audio taper)
    bool   otSat = true;        // output-transformer core saturation
    bool   speakerLoad = true;  // true: speaker impedance curve; false: flat 16 ohm (needs prepare() again)
};

class Core {
public:
    Core();
    // Build topology, matrices for the given solver rate and parameters, solve DC operating point.
    void prepare(double solverRate, const Params& p);
    // Change knob values (rebuilds matrices, ~0.1 ms; state is kept).
    void setParams(const Params& p);
    // One solver-rate sample. vin: volts at the guitar jack (before 7k pickup source resistance).
    // vret: reverb return volts (before level control). Returns speaker voltage.
    double process(double vin, double vret);
    double sendVolts() const { return sendV_; }      // V2a plate: reverb driver input (DC + AC)
    double supplyVolts() const { return bpV_; }
    double lastFlux() const { return lam_; }
    int    lastIterations() const { return lastIts_; }
    long   failCount() const { return fails_; }
    long   fallbackCount() const { return statFallback; }
    bool   dcOk() const { return dcOk_; }
    double nodeV(const std::string& name) const;     // debug: node voltage by name
    int    nodeCount() const { return n_; }
    void   resetStates();                            // back to DC operating point
    void   setFullNewton(bool f) { kFullNewton = f; }
    void   tune2(double contract, double firstStep) { kContract = contract; kFirstStep = firstStep; }
    void   tune(int refactorEvery, double tol, bool extrapolate) { kRefactorEvery = refactorEvery; tolScale_ = tol / 1e-6; kExtrapolate = extrapolate; factorAge_ = 1000; setTol(); }
    void   setTol();

private:
    struct Term { int node; int fixed; };            // node >=0, else fixed rail index
    struct Res { int a, b; double val; int tag; };
    struct Cap { int a, b; double val; int tag; };
    struct Tri { int p, g, k; int mult; };
    struct Pen { int p, g1, g2, k; };
    struct Ind { std::vector<std::pair<int,int>> br; std::vector<double> L; };
    enum Tag { T_NONE, T_GAIN_TOP, T_GAIN_BOT, T_TRE_A, T_TRE_B, T_BASS, T_MID, T_VOL_TOP, T_VOL_BOT,
               T_PRES, T_FBR, T_SLOPE, T_SAG, T_CB };
    enum { F_GND = 0, F_B1, F_B2, F_BM, F_VOC, F_COUNT };

    // topology
    int node(const std::string& s);
    int term(const std::string& s);                  // encoded: >=0 node, <0 -(fixed+1)
    void buildTopology(const Params& p);
    void applyParams(const Params& p);
    void assemble();                                  // matrices from current values
    struct Port { int ta, tb; };                      // port voltage = V(ta)-V(tb), encoded terms
    void newtonPorts(const std::vector<double>& v0, std::vector<double>& v, std::vector<double>& i,
                     const std::vector<double>& Z, bool useSat, bool dcMode, int maxit, double tol, bool ls, int& its, bool& ok);
    void evalDevices(const std::vector<double>& v, std::vector<double>& i, std::vector<double>& D, bool useSat) const;
    void dcSolve();
    void refactor();
    void evalFast(const double* v, double* i);
    void newtonChord(const std::vector<double>& v0, int& its, bool& ok);

    double fs_ = 96000.0, dt_ = 1.0 / 96000.0;
    Params P_;
    bool topoBuilt_ = false;
    std::vector<std::string> names_;
    std::vector<Res> R_; std::vector<Cap> C_; std::vector<Tri> T_; std::vector<Pen> PT_; std::vector<Ind> IND_;
    double rail_[F_COUNT] = {0, 258.0, 272.0, -16.0, 400.0};
    int n_ = 0, nc_ = 0, nb_ = 0, np_ = 0, ldN_ = 0, ldP_ = 0;   // ld*: dimensions padded to a multiple of 4 (SIMD)
    int iin_ = 0, ig4_ = 0, ish_ = 0, ibp_ = 0, ip3_ = 0, ib348_ = 0;
    std::vector<Port> ports_;
    std::vector<double> portConst_;                   // constant part (fixed rails) of port voltages
    int satPort_ = -1;
    // matrices (row-major)
    std::vector<double> H_, HC_, Z_, bt_;              // H_: column-major N x N; HC_: column-major N x np; Z_: row-major np x np
    std::vector<double> Zc_;                            // Z column-major (np x np)
    std::vector<double> invTol_;                         // per-port 1/tolerance (volts) for the convergence test
    double tolScale_ = 3.0;
    std::vector<double> dT_, dP_; double dS_ = 0.0;      // compact device derivatives (triode a,c,d; pentode 3x3; sat)
    std::vector<double> LU_; std::vector<int> piv_;     // chord-Newton factorisation of I + Z D
public: long statRefactor = 0, statEval = 0, statFallback = 0;
private:
    int kRefactorEvery = 8, kMaxIt = 30; double kTol = 1e-6; bool kExtrapolate = true; bool kFullNewton = false; double kContract = 0.25, kFirstStep = 50000.0;
    int factorAge_ = 1000; double vPrev1_[32] = {0}, vPrev2_[32] = {0};
    std::vector<double> Yind_;                         // nb x nb
    std::vector<double> gc_;                           // per-cap companion conductance
    std::vector<int> capA_, capB_;                     // encoded terminals
    std::vector<std::pair<int,int>> brAll_;            // encoded terminals
    // DC
    std::vector<double> Hdc_, HCdc_, Zdc_, bdc_;
    // states
    std::vector<double> x_, cv_, ci_, bv_, bi_, v_, iPort_;
    double lam_ = 0, vsat_ = 0, lam0_ = 0;
    double sendV_ = 0, bpV_ = 0;
    int lastIts_ = 0; long fails_ = 0; bool dcOk_ = false;
    // scratch
    std::vector<double> b_, y_, v0_, D_, J_, dv_, r_;
};

}  // namespace astro
