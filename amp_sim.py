"""Astroverb 16: complete amp prototype (preamp + phase inverter + 2xEL84 + output transformer +
global feedback with presence + speaker load) as one nonlinear network.

Method: modified nodal analysis, trapezoidal capacitors/inductors, Newton per sample.
Also provides a small-signal AC analysis around the DC operating point (fast, exact for the linearised
network) which is used to check frequency response, feedback depth and stability margins.

Deferred / not modelled: supply sag, reverb, OT core saturation and stray capacitance, V5/EL84 grid
blocking details beyond a simple grid-current model.
"""
import math
import numpy as np
from preamp_sim import (FS, DT, CGP, CGK, CPK, PARAMS as PRE_PARAMS, ip_f, ig_f, audio, stamp, mock_signal)

# ---- EL84 (generic, fitted by el84_fit.py) ----
EL_MU, EL_EX, EL_KG1, EL_KP, EL_KVB, EL_KG2 = 20.344, 1.100, 335.87, 123.23, 47.16, 2117.4
EL_GG, EL_CG, EL_XI = 1.0e-3, 12.0, 1.2            # grid-1 current (estimated)
EL_CG1P, EL_CG1K, EL_CPK = 0.5e-12, 10.3e-12, 6.0e-12

PARAMS = dict(PRE_PARAMS)
PARAMS.update(
    FB_R=10e3,            # feedback resistor (drawing: 10k; Atomic 16: 47k)
    PRES=0.5,             # presence 0..1 (1 = least high-frequency feedback = brightest)
    BIAS=-16.0,           # B- volts at the EL84 grid leaks (drawing annotates -16 V)
    SPK="speaker",        # "speaker" (impedance curve) or "resistor" (flat 16 ohm)
    POL=1.0,              # output-transformer secondary polarity (sets feedback sign)
    SAG=0.75,             # 0..1: rectifier/transformer source resistance 25 ohm * 16**SAG (0.75 -> ~200 ohm)
    VOC=None,             # open-circuit supply volts; None = calibrate so idle B+ is 360 V at default bias
    C_FILT1=47e-6, C_FILT2=33e-6,   # B+ reservoir and screen/preamp-feed reservoir (from the drawing)
    I_PRE=7.6e-3,         # constant draw of preamp + PI + reverb tube through the 10k/3W dropper (calibrated
                          # so the 348 V node matches the drawing; preamp rails themselves stay ideal)
    C_OT=1e-9,          # OT winding capacitance: each plate to ground, and 2x that across the full primary
    OT_SAT=True,          # output-transformer core saturation on/off
    LAM_K=1.4,            # flux-linkage knee (V*s); about where the core starts to saturate near 70 Hz at full power
    LPP=20.0,             # primary inductance plate-to-plate, henries (generic guess)
    RAA=8000.0, ZL=16.0,  # plate-to-plate load, speaker nominal
    K_PS=0.9993, K12=0.9998,   # coupling: primary-secondary (leakage ~28 mH), half-to-half
    R_PRI=70.0, R_SEC=0.5,     # winding DC resistances (generic guesses)
    RE=13.0, LE=1.0e-3, F0=100.0, QMS=5.0, RM=45.0,   # speaker: voice coil + resonance
)


def el_i(vpk, vg1k, vg2k):
    if vg2k < 1.0:
        vg2k = 1.0
    x = EL_KP * (1.0 / EL_MU + vg1k / vg2k)
    sp = x if x > 30.0 else math.log1p(math.exp(x))
    e1 = vg2k / EL_KP * sp
    if e1 <= 0.0:
        return 0.0, 0.0
    f = 2.0 * e1 ** EL_EX
    ip = f / EL_KG1 * math.atan(vpk / EL_KVB) if vpk > 0.0 else 0.0
    return ip, f / EL_KG2


def el_ig(vg1k):
    y = EL_CG * vg1k
    s = y if y > 30.0 else math.log1p(math.exp(y))
    return EL_GG * (s / EL_CG) ** EL_XI


def stamp_Y(G, b, branches, Y):
    """Stamp a multi-branch admittance block (current leaving a_k = sum_l Y[k,l]*(Va_l - Vb_l))."""
    for k, (ak, bk) in enumerate(branches):
        for l, (al, bl) in enumerate(branches):
            y = Y[k, l]
            for rk, sr in ((ak, 1.0), (bk, -1.0)):
                if rk[0] < 0:
                    continue
                for cn, sc in ((al, 1.0), (bl, -1.0)):
                    coef = sr * sc * y
                    if cn[0] >= 0:
                        G[rk[0], cn[0]] += coef
                    else:
                        b[rk[0]] -= coef * cn[1]


class Amp:
    def __init__(self, **kw):
        P = dict(PARAMS); P.update(kw); self.P = P
        rsrc = 25.0 * 16.0 ** P["SAG"]
        voc = P["VOC"] if P["VOC"] is not None else supply_voc(P["SAG"])
        FIX = {"gnd": 0.0, "B1": 258.0, "B2": 272.0, "BM": P["BIAS"], "VOC": voc}
        self.FIX = FIX
        R, C, T, PT, IND = [], [], [], [], []

        def r(a, b, v): R.append((a, b, max(v, 1.0)))
        def c(a, b, v): C.append((a, b, v))
        def tri(p, g, k):
            T.append((p, g, k)); c(g, p, CGP); c(g, k, CGK); c(p, k, CPK)
        def pent(p, g1, g2, k):
            PT.append((p, g1, g2, k)); c(g1, p, EL_CG1P); c(g1, k, EL_CG1K); c(p, k, EL_CPK)
        def ind(branches, L): IND.append((branches, np.array(L, dtype=float)))

        # ---------------- preamp (same as preamp_sim, minus the PI-return change) ----------------
        r("in", "gnd", P["RS"]); r("in", "gnd", 1e6); r("in", "g1", 69e3)
        tri("p1", "g1", "k1"); r("p1", "B1", 220e3); r("k1", "gnd", 1.8e3); c("k1", "gnd", 1e-6)
        c("p1", "a0", 22e-9); r("a0", "a1", 1e6)
        ga = P["GAIN_F"] if P["GAIN_F"] is not None else (P["GAIN_K"] ** P["GAIN"] - 1.0) / (P["GAIN_K"] - 1.0)
        r("a1", "w1", 1e6 * (1 - ga)); r("w1", "gnd", 1e6 * ga)
        tri("p2", "w1", "k2"); r("p2", "B1", 100e3); r("k2", "gnd", 2.2e3)
        c("p2", "b0", 22e-9); r("b0", "b1", 1e6); c("b0", "b1", P["CB"])
        r("b1", "g3", 220e3); r("g3", "gnd", 330e3)
        tri("p3", "g3", "k3"); r("p3", "B2", P["V2A_RP"]); r("k3", "gnd", 1.8e3); c("k3", "gnd", 1e-6)
        c("p3", "h0", 22e-9); r("h0", "g4", 680e3); r("g4", "gnd", 15e3)
        tri("p4", "g4", "k4"); tri("p4", "g4", "k4")      # V2b + V3a: grids tied (net "C", reverb return mixes here)
        r("p4", "B2", 100e3); r("k4", "gnd", 1e3); c("k4", "gnd", 100e-6)
        tri("B2", "p4", "k5"); r("k5", "gnd", P["V3B_RK"])
        rox = 250e3 * audio(P["TREBLE"])
        c("k5", "pt", 470e-12); r("pt", "o", 250e3 - rox); r("o", "x", rox)
        r("k5", "l", P["RSLOPE"]); c("l", "x", 22e-9); c("l", "y", 22e-9)
        r("x", "y", 1e6 * audio(P["BASS"])); r("y", "gnd", 25e3 * P["MID"])
        r("o", "v", P["R_SER"]); r("v", "gnd", P["R_SH"])
        vg = audio(P["VOL"])
        r("v", "wv", 1e6 * (1 - vg)); r("wv", "gnd", 1e6 * vg)
        c("wv", "g6", 22e-9); r("g6", "t0", 1e6)

        # ---------------- phase inverter V5 and its feedback return ----------------
        tri("pa", "g6", "ktail"); tri("pb", "gb", "ktail")
        r("pa", "B2", 82e3); r("pb", "B2", 100e3); c("pa", "pb", 120e-12)
        r("ktail", "t0", 470.0)
        r("gb", "t0", 1e6)
        r("t0", "nn", 10e3); r("nn", "gnd", 4.7e3)          # Atomic-style return: lifts PI cathode to ~29 V
        r("nn", "sh", P["FB_R"]); c("nn", "gb", 100e-9)
        c("nn", "q", 100e-9); r("q", "gnd", 25e3 * audio(1.0 - P["PRES"]))   # presence shunt, audio-style law

        # ---------------- output stage ----------------
        c("pa", "gv6", 22e-9); c("pb", "gv7", 22e-9)
        r("gv6", "BM", 220e3); r("gv7", "BM", 220e3)
        r("gv6", "g16", 2.2e3); r("gv7", "g17", 2.2e3)
        pent("op1", "g16", "s6", "gnd"); pent("op2", "g17", "s7", "gnd")
        r("s6", "b348", 470.0); r("s7", "b348", 470.0)

        # supply: source (rectifier + transformer) -> B+ reservoir -> 1k -> 348 V node (screens, preamp feed)
        r("bp", "VOC", rsrc); c("bp", "gnd", P["C_FILT1"])
        r("bp", "b348", 1e3); c("b348", "gnd", P["C_FILT2"])

        # output transformer: halves wound along end1 -> CT -> end2, secondary hot -> cold(gnd)
        c("op1", "gnd", P["C_OT"]); c("op2", "gnd", P["C_OT"]); c("op1", "op2", 2 * P["C_OT"])
        r("op1", "w1a", P["R_PRI"]); r("w2b", "op2", P["R_PRI"]); r("shw", "sh", P["R_SEC"])
        Lh = P["LPP"] / 4.0
        n = math.sqrt(P["RAA"] / P["ZL"])
        Ls = Lh * (2.0 / n) ** 2
        m = P["POL"] * P["K_PS"] * math.sqrt(Lh * Ls)
        Lot = [[Lh, P["K12"] * Lh, m], [P["K12"] * Lh, Lh, m], [m, m, Ls]]
        ind([("w1a", "bp"), ("bp", "w2b"), ("shw", "gnd")], Lot)

        # speaker load
        if P["SPK"] == "speaker":
            r("sh", "s1", P["RE"]); ind([("s1", "s2")], [[P["LE"]]])
            w0 = 2 * math.pi * P["F0"]
            Lm = P["RM"] / (w0 * P["QMS"]); Cm = 1.0 / (w0 * w0 * Lm)
            r("s2", "gnd", P["RM"]); ind([("s2", "gnd")], [[Lm]]); c("s2", "gnd", Cm)
        else:
            r("sh", "gnd", P["ZL"])

        # ---------------- assemble ----------------
        names = []
        def reg(nm):
            if nm not in FIX and nm not in names: names.append(nm)
        for e in R + C:
            reg(e[0]); reg(e[1])
        for t in T + PT:
            for nm in t: reg(nm)
        for br, _ in IND:
            for a, b in br: reg(a); reg(b)
        self.names = names; self.idx = {nm: i for i, nm in enumerate(names)}
        N = self.n = len(names)
        term = lambda nm: (-1, FIX[nm]) if nm in FIX else (self.idx[nm], 0.0)
        self.term = term

        G = np.zeros((N, N)); b = np.zeros(N)
        for a, bn, v in R:
            stamp(G, b, term(a), term(bn), 1.0 / v)
        G += 1e-12 * np.eye(N)
        self.Gres, self.bres = G.copy(), b.copy()

        # capacitors
        nc = len(C)
        self.ia = np.zeros(nc, int); self.ib = np.zeros(nc, int)
        self.fa = np.zeros(nc); self.fb = np.zeros(nc)
        self.gc = np.zeros(nc); self.Mcap = np.zeros((N, nc)); self.Ccap = np.zeros((N, N))
        Gt, bt = G.copy(), b.copy()
        for j, (a, bn, v) in enumerate(C):
            ta, tb = term(a), term(bn)
            self.ia[j], self.fa[j] = ta; self.ib[j], self.fb[j] = tb
            self.gc[j] = 2.0 * v / DT
            stamp(Gt, bt, ta, tb, self.gc[j])
            stamp(self.Ccap, np.zeros(N), ta, tb, v)
            if ta[0] >= 0: self.Mcap[ta[0], j] = 1.0
            if tb[0] >= 0: self.Mcap[tb[0], j] = -1.0

        # inductor branches (global list, block-diagonal matrices)
        br_all = []; sizes = []
        for br, _ in IND:
            br_all += [(term(a), term(b)) for a, b in br]; sizes.append(len(br))
        nb = len(br_all); self.nb = nb
        Lg = np.zeros((nb, nb)); o = 0
        for (br, Lm_), s in zip(IND, sizes):
            Lg[o:o + s, o:o + s] = Lm_; o += s
        self.Linv = np.linalg.inv(Lg)
        self.Yind = (DT / 2.0) * self.Linv
        self.br_all = br_all
        stamp_Y(Gt, bt, br_all, self.Yind)
        self.Mind = np.zeros((N, nb))
        for k, (ak, bk) in enumerate(br_all):
            if ak[0] >= 0: self.Mind[ak[0], k] = -1.0
            if bk[0] >= 0: self.Mind[bk[0], k] = 1.0
        self.Gind_inv = np.zeros((N, N))
        stamp_Y(self.Gind_inv, np.zeros(N), br_all, self.Linv)
        self.Gt, self.bt = Gt, bt
        # DC matrix: inductors are shorts
        Gdc, bdc = G.copy(), b.copy()
        for ak, bk in br_all:
            stamp(Gdc, bdc, ak, bk, 1e5)
        self.Gdc, self.bdc = Gdc, bdc

        self.tubes = [("t", [term(x) for x in t]) for t in T] + [("p", [term(x) for x in t]) for t in PT]
        self.iin = self.idx["in"]
        # constant current drawn through the 10k/3W dropper by the (ideal-rail) preamp, PI and reverb tube
        i348 = self.idx["b348"]
        self.bt[i348] -= P["I_PRE"]; self.bdc[i348] -= P["I_PRE"]
        # output-transformer core: nonlinear magnetizing branch across the whole primary (flux-dependent)
        self.sat = [(term("w1a"), term("w2b"))] if P["OT_SAT"] else []
        self.sat_on = False; self.lam0 = 0.0

    # ---- nonlinear residual / Jacobian ----
    def FJ(self, G, b, x, jac=True):
        F = G @ x - b
        J = G.copy() if jac else None
        e = 1e-3
        for kind, terms in self.tubes:
            vals = [x[i] if i >= 0 else f for i, f in terms]
            if kind == "t":
                vp, vg, vk = vals
                vpk, vgk = vp - vk, vg - vk
                i_p, i_g = ip_f(vpk, vgk), ig_f(vgk)
                cur = [i_p, i_g, -(i_p + i_g)]
                if jac:
                    a = (ip_f(vpk + e, vgk) - i_p) / e
                    c = (ip_f(vpk, vgk + e) - i_p) / e
                    d = (ig_f(vgk + e) - i_g) / e
                    dI = [[a, c, -(a + c)], [0.0, d, -d], [-a, -(c + d), a + c + d]]
            else:
                vp, vg1, vg2, vk = vals
                vpk, v1, v2 = vp - vk, vg1 - vk, vg2 - vk
                i_p, i_2 = el_i(vpk, v1, v2)
                i_1 = el_ig(v1)
                cur = [i_p, i_1, i_2, -(i_p + i_1 + i_2)]
                if jac:
                    ap = (el_i(vpk + e, v1, v2)[0] - i_p) / e
                    c1 = (el_i(vpk, v1 + e, v2)[0] - i_p) / e
                    c2 = (el_i(vpk, v1, v2 + e)[0] - i_p) / e
                    d1 = (el_ig(v1 + e) - i_1) / e
                    e1 = (el_i(vpk, v1 + e, v2)[1] - i_2) / e
                    e2 = (el_i(vpk, v1, v2 + e)[1] - i_2) / e
                    rp = [ap, c1, c2, -(ap + c1 + c2)]
                    rg1 = [0.0, d1, 0.0, -d1]
                    rg2 = [0.0, e1, e2, -(e1 + e2)]
                    rk = [-(rp[j] + rg1[j] + rg2[j]) for j in range(4)]
                    dI = [rp, rg1, rg2, rk]
            for i, (ti, _) in enumerate(terms):
                if ti >= 0:
                    F[ti] += cur[i]
                    if jac:
                        for j, (tj, _) in enumerate(terms):
                            if tj >= 0:
                                J[ti, tj] += dI[i][j]
        if self.sat and self.sat_on:
            Lp, lk, h = self.P["LPP"], self.P["LAM_K"], 0.5 * DT
            for (ta, tb) in self.sat:
                va = x[ta[0]] if ta[0] >= 0 else ta[1]
                vb = x[tb[0]] if tb[0] >= 0 else tb[1]
                lam = self.lam0 + h * (va - vb)
                lc = max(-4.0 * lk, min(4.0 * lk, lam))
                i_s = (lc / Lp) * (lc / lk) ** 4                  # extra magnetizing current, ~0 until near the knee
                dg = h * 5.0 * lc ** 4 / (Lp * lk ** 4)
                if ta[0] >= 0:
                    F[ta[0]] += i_s
                    if jac:
                        J[ta[0], ta[0]] += dg
                        if tb[0] >= 0: J[ta[0], tb[0]] -= dg
                if tb[0] >= 0:
                    F[tb[0]] -= i_s
                    if jac:
                        J[tb[0], tb[0]] += dg
                        if ta[0] >= 0: J[tb[0], ta[0]] -= dg
        return F, J

    def sat_v(self, x):
        if not self.sat:
            return 0.0
        (ia, fa), (ib, fb) = self.sat[0]
        return (x[ia] if ia >= 0 else fa) - (x[ib] if ib >= 0 else fb)

    def newton(self, G, b, x, maxit=30, tol=1e-6, ls=False):
        for it in range(maxit):
            F, J = self.FJ(G, b, x)
            dx = np.linalg.solve(J, -F)
            mx = np.max(np.abs(dx))
            if mx > 25.0:
                dx *= 25.0 / mx
            a = 1.0
            if ls:
                n0 = np.linalg.norm(F)
                for _ in range(16):
                    Fn, _ = self.FJ(G, b, x + a * dx, jac=False)
                    if np.linalg.norm(Fn) < n0 or n0 < 1e-13:
                        break
                    a *= 0.5
            x = x + a * dx
            if np.max(np.abs(a * dx)) < tol:
                return x, it + 1, True
        return x, maxit, False

    def V(self, x, nm):
        return self.FIX[nm] if nm in self.FIX else x[self.idx[nm]]

    def dc(self):
        self.sat_on = False
        x, its, ok = self.newton(self.Gdc, self.bdc, np.zeros(self.n), maxit=400, tol=1e-8, ls=True)
        self.x_dc, self.dc_ok, self.dc_its = x, ok, its
        P = self.P
        i0 = np.zeros(self.nb)
        i0[0] = (self.V(x, "op1") - self.V(x, "w1a")) / P["R_PRI"]
        i0[1] = (self.V(x, "w2b") - self.V(x, "op2")) / P["R_PRI"]
        i0[2] = (self.V(x, "sh") - self.V(x, "shw")) / P["R_SEC"]
        self.i_ind0 = i0
        return x, ok

    def capv(self, x):
        xa = np.where(self.ia >= 0, x[np.maximum(self.ia, 0)], self.fa)
        xb = np.where(self.ib >= 0, x[np.maximum(self.ib, 0)], self.fb)
        return xa - xb

    def indv(self, x):
        out = np.empty(self.nb)
        for k, ((ia, fa), (ib, fb)) in enumerate(self.br_all):
            out[k] = (x[ia] if ia >= 0 else fa) - (x[ib] if ib >= 0 else fb)
        return out

    def run(self, vs, record):
        x = self.x_dc.copy()
        cv = self.capv(x); ci = np.zeros_like(cv)
        bv = self.indv(x); bi = self.i_ind0.copy()
        self.sat_on = True
        lam = 0.0; vsat = self.sat_v(x); lam_log = np.empty(len(vs))
        ridx = [self.idx[nm] for nm in record]
        out = np.empty((len(record), len(vs)))
        fails = 0; tot = 0
        for k, v in enumerate(vs):
            ieq = self.gc * cv + ci
            ih = bi + self.Yind @ bv
            b = self.bt + self.Mcap @ ieq + self.Mind @ ih
            b[self.iin] += v / self.P["RS"]
            self.lam0 = lam + 0.5 * DT * vsat
            x, its, ok = self.newton(self.Gt, b, x)
            fails += (not ok); tot += its
            vnew = self.sat_v(x); lam = self.lam0 + 0.5 * DT * vnew; vsat = vnew; lam_log[k] = lam
            cvn = self.capv(x)
            ci = self.gc * cvn - ieq; cv = cvn
            bvn = self.indv(x)
            bi = bi + self.Yind @ (bv + bvn); bv = bvn
            out[:, k] = x[ridx]
        self.sat_on = False; self.last_lam = lam_log
        return out, fails, tot / len(vs)

    # ---- small-signal AC about the DC operating point ----
    def ac(self, freqs, nodes):
        _, Jnl = self.FJ(np.zeros((self.n, self.n)), np.zeros(self.n), self.x_dc)
        base = self.Gres + Jnl - np.zeros((self.n, self.n))
        # FJ with G=0 returns only nonlinear derivatives; Gres adds the resistors
        e = np.zeros(self.n, dtype=complex); e[self.iin] = 1.0 / self.P["RS"]
        ridx = [self.idx[nm] for nm in nodes]
        out = np.empty((len(nodes), len(freqs)), dtype=complex)
        for k, f in enumerate(freqs):
            w = 2 * math.pi * f
            A = base + 1j * w * self.Ccap + self.Gind_inv / (1j * w)
            out[:, k] = np.linalg.solve(A, e)[ridx]
        return out


_VOC_CACHE = {}


def supply_voc(sag):
    """Open-circuit supply voltage that gives 360 V at B+ at idle (default bias) for this SAG setting."""
    key = round(float(sag), 4)
    if key not in _VOC_CACHE:
        rsrc = 25.0 * 16.0 ** key
        voc = 400.0
        for _ in range(6):
            a = Amp(VOC=voc, SAG=key)
            x, ok = a.dc()
            voc = 360.0 + rsrc * (voc - a.V(x, "bp")) / rsrc
        _VOC_CACHE[key] = voc
    return _VOC_CACHE[key]
