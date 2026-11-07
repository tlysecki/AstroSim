"""Astroverb 16 preamp (V1a..V3b + tone stack + volume): full-network transient prototype.

Method: modified nodal analysis, trapezoidal capacitors, Newton iteration per sample.
Triodes: Koren plate current + Dempwolf-style grid current + Miller/interelectrode caps.
Rails are ideal DC (supply sag, reverb, power amp, output transformer: deferred).

Resistor values changed from the drawing (see PARAMS comments) were chosen to match the
printed voltages / the Atomic 16 schematic.
"""
import math
import numpy as np

FS = 96000.0
DT = 1.0 / FS

# ---- tube model (12AX7) ----
MU, EX, KG1, KP, KVB = 100.0, 1.4, 1060.0, 600.0, 300.0
GG, CG, XI = 6.06e-4, 11.8, 1.2          # grid current: estimated, not fitted to a real tube
CGP, CGK, CPK = 1.7e-12, 1.6e-12, 0.46e-12

FIXED = {"gnd": 0.0, "B1": 258.0, "B2": 272.0}

PARAMS = dict(
    GAIN=0.7, BASS=0.5, MID=0.5, TREBLE=0.5, VOL=0.7,   # pot rotations 0..1
    GAIN_K=5871.0,          # gain-knob law: pot fraction = (K**x - 1)/(K - 1); 100 = standard audio taper.
                            # 12750 puts the level plateau at knob 0.65 (calibrate_gain.py, 0.25 V ref input)
    GAIN_F=None,            # if set, overrides the knob law with a raw pot fraction (calibration only)
    RS=7e3,                 # guitar source resistance (pickup), reactive part ignored
    V2A_RP=100e3,           # drawing: 220k   (override: matches 187 V / 1.5 V, Atomic 100K)
    V3B_RK=100e3,           # drawing: 1k     (override: matches 153 V, Atomic 100K)
    RSLOPE=47e3,            # drawing: 1k     (override: Atomic 47K; 1k kills the tone controls)
    R_SER=470e3, R_SH=150e3,  # drawing: 1k/1k (override: Atomic 470K/150K)
    CB=470e-12,             # drawing: 22nF across the 1M (SUSPECT, see notes); Atomic: 470p
)


def audio(x):
    return (100.0 ** x - 1.0) / 99.0


def ip_f(vpk, vgk):
    x = KP * (1.0 / MU + vgk / math.sqrt(KVB + vpk * vpk))
    e1 = vpk / KP * (x if x > 30.0 else math.log1p(math.exp(x)))
    return 0.0 if e1 <= 0.0 else 2.0 * e1 ** EX / KG1


def ig_f(vgk):
    y = CG * vgk
    s = y if y > 30.0 else math.log1p(math.exp(y))
    return GG * (s / CG) ** XI


def stamp(G, b, ta, tb, g):
    ia, va = ta
    ib, vb = tb
    if ia >= 0:
        G[ia, ia] += g
        if ib >= 0:
            G[ia, ib] -= g
        else:
            b[ia] += g * vb
    if ib >= 0:
        G[ib, ib] += g
        if ia >= 0:
            G[ib, ia] -= g
        else:
            b[ib] += g * va


class Preamp:
    def __init__(self, **kw):
        P = dict(PARAMS)
        P.update(kw)
        self.P = P
        R, C, T = [], [], []

        def r(a, b, v):
            R.append((a, b, max(v, 1.0)))

        def c(a, b, v):
            C.append((a, b, v))

        def tri(p, g, k):
            T.append((p, g, k))
            c(g, p, CGP); c(g, k, CGK); c(p, k, CPK)

        # input
        r("in", "gnd", P["RS"]); r("in", "gnd", 1e6); r("in", "g1", 69e3)
        # V1a
        tri("p1", "g1", "k1"); r("p1", "B1", 220e3); r("k1", "gnd", 1.8e3); c("k1", "gnd", 1e-6)
        c("p1", "a0", 22e-9); r("a0", "a1", 1e6)
        ga = P["GAIN_F"] if P["GAIN_F"] is not None else (P["GAIN_K"] ** P["GAIN"] - 1.0) / (P["GAIN_K"] - 1.0)
        r("a1", "w1", 1e6 * (1 - ga)); r("w1", "gnd", 1e6 * ga)
        # V1b
        tri("p2", "w1", "k2"); r("p2", "B1", 100e3); r("k2", "gnd", 2.2e3)
        c("p2", "b0", 22e-9); r("b0", "b1", 1e6); c("b0", "b1", P["CB"])
        r("b1", "g3", 220e3); r("g3", "gnd", 330e3)
        # V2a
        tri("p3", "g3", "k3"); r("p3", "B2", P["V2A_RP"]); r("k3", "gnd", 1.8e3); c("k3", "gnd", 1e-6)
        c("p3", "h0", 22e-9); r("h0", "g4", 680e3); r("g4", "gnd", 15e3)
        # V2b + V3a (shared plate/cathode; V3a grid = reverb return, grounded here)
        tri("p4", "g4", "k4"); tri("p4", "g4", "k4")      # V2b + V3a grids tied (net C)
        r("p4", "B2", 100e3); r("k4", "gnd", 1e3); c("k4", "gnd", 100e-6)
        # V3b cathode follower
        tri("B2", "p4", "k5"); r("k5", "gnd", P["V3B_RK"])
        # tone stack
        rox = 250e3 * audio(P["TREBLE"])
        c("k5", "pt", 470e-12); r("pt", "o", 250e3 - rox); r("o", "x", rox)
        r("k5", "l", P["RSLOPE"]); c("l", "x", 22e-9); c("l", "y", 22e-9)
        r("x", "y", 1e6 * audio(P["BASS"])); r("y", "gnd", 25e3 * P["MID"])
        # output network + volume + V5 grid coupling
        r("o", "v", P["R_SER"]); r("v", "gnd", P["R_SH"])
        vg = audio(P["VOL"])
        r("v", "wv", 1e6 * (1 - vg)); r("wv", "gnd", 1e6 * vg)
        c("wv", "g6", 22e-9); r("g6", "gnd", 1e6)

        names = []
        for e in R + C:
            for nme in e[:2]:
                if nme not in FIXED and nme not in names:
                    names.append(nme)
        for t in T:
            for nme in t:
                if nme not in FIXED and nme not in names:
                    names.append(nme)
        self.names = names
        self.idx = {nme: i for i, nme in enumerate(names)}
        n = self.n = len(names)

        def term(nme):
            return (-1, FIXED[nme]) if nme in FIXED else (self.idx[nme], 0.0)

        G = np.zeros((n, n)); b = np.zeros(n)
        for a, bn, v in R:
            stamp(G, b, term(a), term(bn), 1.0 / v)
        G += 1e-12 * np.eye(n)
        self.Gdc, self.bdc = G.copy(), b.copy()

        Gt, bt = G.copy(), b.copy()
        nc = len(C)
        self.ia = np.zeros(nc, int); self.ib = np.zeros(nc, int)
        self.fa = np.zeros(nc); self.fb = np.zeros(nc)
        self.gc = np.zeros(nc); self.M = np.zeros((n, nc))
        for j, (a, bn, v) in enumerate(C):
            ta, tb = term(a), term(bn)
            self.ia[j], self.fa[j] = ta; self.ib[j], self.fb[j] = tb
            self.gc[j] = 2.0 * v / DT
            stamp(Gt, bt, ta, tb, self.gc[j])
            if ta[0] >= 0: self.M[ta[0], j] = 1.0
            if tb[0] >= 0: self.M[tb[0], j] = -1.0
        self.Gt, self.bt = Gt, bt
        self.tri = [term(p) + term(g) + term(k) for p, g, k in T]
        self.iin = self.idx["in"]

    # nonlinear residual / Jacobian
    def FJ(self, G, b, x, jac=True):
        F = G @ x - b
        J = G.copy() if jac else None
        for (ip, fp), (ig, fg), (ik, fk) in [((t[0], t[1]), (t[2], t[3]), (t[4], t[5])) for t in self.tri]:
            vp = x[ip] if ip >= 0 else fp
            vg = x[ig] if ig >= 0 else fg
            vk = x[ik] if ik >= 0 else fk
            vpk, vgk = vp - vk, vg - vk
            i_p, i_g = ip_f(vpk, vgk), ig_f(vgk)
            if ip >= 0: F[ip] += i_p
            if ig >= 0: F[ig] += i_g
            if ik >= 0: F[ik] -= i_p + i_g
            if jac:
                e = 1e-3
                a = (ip_f(vpk + e, vgk) - i_p) / e
                c = (ip_f(vpk, vgk + e) - i_p) / e
                d = (ig_f(vgk + e) - i_g) / e
                if ip >= 0:
                    J[ip, ip] += a
                    if ig >= 0: J[ip, ig] += c
                    if ik >= 0: J[ip, ik] -= a + c
                if ig >= 0:
                    J[ig, ig] += d
                    if ik >= 0: J[ig, ik] -= d
                if ik >= 0:
                    if ip >= 0: J[ik, ip] -= a
                    if ig >= 0: J[ik, ig] -= c + d
                    J[ik, ik] += a + c + d
        return F, J

    def newton(self, G, b, x, maxit=30, tol=1e-6, ls=False):
        for it in range(maxit):
            F, J = self.FJ(G, b, x)
            dx = np.linalg.solve(J, -F)
            m = np.max(np.abs(dx))
            if m > 20.0:
                dx *= 20.0 / m
            a = 1.0
            if ls:
                n0 = np.linalg.norm(F)
                for _ in range(14):
                    Fn, _ = self.FJ(G, b, x + a * dx, jac=False)
                    if np.linalg.norm(Fn) < n0 or n0 < 1e-13:
                        break
                    a *= 0.5
            x = x + a * dx
            if np.max(np.abs(a * dx)) < tol:
                return x, it + 1, True
        return x, maxit, False

    def dc(self):
        x, its, ok = self.newton(self.Gdc, self.bdc, np.zeros(self.n), maxit=300, tol=1e-8, ls=True)
        self.x_dc, self.dc_ok = x, ok
        return x, ok

    def capv(self, x):
        xa = np.where(self.ia >= 0, x[np.maximum(self.ia, 0)], self.fa)
        xb = np.where(self.ib >= 0, x[np.maximum(self.ib, 0)], self.fb)
        return xa - xb

    def run(self, vs, record):
        x = self.x_dc.copy()
        cv = self.capv(x); ci = np.zeros_like(cv)
        ridx = [self.idx[nme] for nme in record]
        out = np.empty((len(record), len(vs)))
        fails = 0; tot_it = 0
        for k, v in enumerate(vs):
            ieq = self.gc * cv + ci
            b = self.bt + self.M @ ieq
            b[self.iin] += v / self.P["RS"]
            x, its, ok = self.newton(self.Gt, b, x)
            fails += (not ok); tot_it += its
            cvn = self.capv(x)
            ci = self.gc * cvn - ieq
            cv = cvn
            out[:, k] = x[ridx]
        return out, fails, tot_it / len(vs)


def mock_signal(kind, peak, seconds=0.15):
    t = np.arange(int(seconds * FS)) / FS
    if kind == "sine":              # 240 Hz: 400 samples/period at 96 kHz
        s = np.sin(2 * np.pi * 240 * t)
    else:                           # pseudo-guitar: 120 Hz + harmonics, 1/n rolloff
        s = sum(np.sin(2 * np.pi * 120 * n * t + 0.7 * n) / n for n in range(1, 41))   # to 4.8 kHz
        s /= np.max(np.abs(s))
    fade = np.minimum(1.0, t / 0.01)
    return peak * s * fade
