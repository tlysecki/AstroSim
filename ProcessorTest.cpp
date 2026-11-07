// Offline test of the plugin processor (no audio device, no GUI): levels, limiter, quality modes, IR, hosted reverb, CPU.
#include "../src/PluginProcessor.h"
#include <cstdio>
#include <chrono>

static double rms(const float* x, int n) { double s = 0; for (int i = 0; i < n; ++i) s += (double) x[i] * x[i]; return std::sqrt(s / n); }

static void pump(int ms) { auto end = juce::Time::getMillisecondCounter() + (juce::uint32) ms; while (juce::Time::getMillisecondCounter() < end) { juce::MessageManager::getInstance()->runDispatchLoopUntil(20); } }
struct Result { double peak, rmsv, cpuPct; int fails; double nan; };

static Result run(AstroverbAudioProcessor& p, double sr, int bs, double seconds, double ampIn, double f0, double warmup = 0.3)
{
    juce::AudioBuffer<float> buf(2, bs); juce::MidiBuffer midi;
    const int blocks = (int) (seconds * sr / bs);
    double ph = 0, peak = 0, sumsq = 0; long cnt = 0; double nan = 0; double procTime = 0;
    for (int b = 0; b < blocks; ++b)
    {
        for (int i = 0; i < bs; ++i)
        {
            ph += 2.0 * juce::MathConstants<double>::pi * f0 / sr;
            const float s = (float) (ampIn * (std::sin(ph) + 0.5 * std::sin(2 * ph) + 0.33 * std::sin(3 * ph)) / 1.83);
            buf.setSample(0, i, s); buf.setSample(1, i, s);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        p.processBlock(buf, midi);
        procTime += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        if (b * bs > warmup * sr)
            for (int i = 0; i < bs; ++i)
            {
                const float v = buf.getSample(0, i);
                if (!std::isfinite(v)) nan += 1; else peak = std::max<double>(peak, std::fabs(v));
                sumsq += (double) v * v; ++cnt;
            }
    }
    return { peak, std::sqrt(sumsq / std::max<long>(cnt, 1)), 100.0 * procTime / (blocks * bs / sr), p.solverFails.load(), nan };
}

static void setParam(AstroverbAudioProcessor& p, const char* id, float real)
{
    auto* prm = p.apvts.getParameter(id); prm->setValueNotifyingHost(prm->convertTo0to1(real));
}

int main(int argc, char** argv)
{
    juce::ScopedJuceInitialiser_GUI init;
    struct Case { const char* name; double sr; int bs; float quality; float gain; float vol; float inTrim; double amp; };
    const Case cases[] = {
        { "48k High   default      ", 48000, 256, 1, 0.65f, 0.7f, 0, 0.25 },
        { "48k High   gain 10 vol 10", 48000, 256, 1, 1.0f, 1.0f, 0, 0.5 },
        { "48k High   hot input +24dB", 48000, 256, 1, 1.0f, 1.0f, 24, 1.0 },
        { "48k Std    gain 10 vol 10", 48000, 256, 0, 1.0f, 1.0f, 0, 0.5 },
        { "44.1k High gain 10 vol 10", 44100, 128, 1, 1.0f, 1.0f, 0, 0.5 },
        { "96k Std    gain 10 vol 10", 96000, 512, 1, 1.0f, 1.0f, 0, 0.5 },
        { "48k High   clean (gain 3)", 48000, 64, 1, 0.3f, 0.7f, 0, 0.1 },
        { "96k Std    clean (gain 3)", 96000, 64, 0, 0.3f, 0.7f, 0, 0.1 },
        { "96k Std    edge (gain 6.5)", 96000, 256, 0, 0.65f, 0.7f, 0, 0.25 },
        { "48k High   edge (gain 6.5)", 48000, 256, 1, 0.65f, 0.7f, 0, 0.25 },
        { "96k Std    gain 10 vol 10 b256", 96000, 256, 0, 1.0f, 1.0f, 0, 0.5 },
    };
    printf("%-30s %8s %8s %8s %6s %6s\n", "case (A2 110 Hz, mock input)", "peak FS", "rms FS", "CPU %", "fails", "NaNs");
    for (auto& c : cases)
    {
        AstroverbAudioProcessor p;
        p.setPlayConfigDetails(2, 2, c.sr, c.bs);
        setParam(p, "gain", c.gain); setParam(p, "volume", c.vol); setParam(p, "quality", c.quality); setParam(p, "intrim", c.inTrim);
        p.prepareToPlay(c.sr, c.bs);
        auto r = run(p, c.sr, c.bs, 1.5, c.amp, 110.0);
        printf("%-30s %8.3f %8.3f %8.1f %6d %6.0f\n", c.name, r.peak, r.rmsv, r.cpuPct, r.fails, r.nan);
    }

    // parameter sweep while playing: no blow-ups, no NaN
    {
        AstroverbAudioProcessor p; p.setPlayConfigDetails(2, 2, 48000, 256); p.prepareToPlay(48000, 256);
        juce::AudioBuffer<float> buf(2, 256); juce::MidiBuffer midi; double ph = 0, peak = 0; int bad = 0;
        const char* ids[] = { "gain", "bass", "mid", "treble", "volume", "presence", "feedback", "sag", "bright", "bias" };
        for (int b = 0; b < 1500; ++b)
        {
            if (b % 20 == 0) for (auto* id : ids) { auto* prm = p.apvts.getParameter(id); prm->setValueNotifyingHost((float) ((b / 20 * 37 + (int) id[0] * 11) % 101) / 100.0f); }
            for (int i = 0; i < 256; ++i) { ph += 2 * juce::MathConstants<double>::pi * 196.0 / 48000.0; float s = (float) (0.3 * std::sin(ph)); buf.setSample(0, i, s); buf.setSample(1, i, s); }
            p.processBlock(buf, midi);
            for (int i = 0; i < 256; ++i) { float v = buf.getSample(0, i); if (!std::isfinite(v)) ++bad; else peak = std::max<double>(peak, std::fabs(v)); }
        }
        printf("random knob sweep (75 changes of every knob): peak %.3f FS, bad samples %d, solver fails %d\n", peak, bad, p.solverFails.load());
    }

    // synthetic IR (decaying noise, 60 ms) through the built-in convolution
    {
        juce::File ir = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("test_ir.wav");
        { juce::WavAudioFormat wav; ir.deleteFile(); std::unique_ptr<juce::FileOutputStream> os(ir.createOutputStream());
          std::unique_ptr<juce::AudioFormatWriter> w(wav.createWriterFor(os.get(), 48000, 1, 24, {}, 0)); if (w) { os.release();
            juce::AudioBuffer<float> b(1, 2880); juce::Random rnd(7);
            for (int i = 0; i < 2880; ++i) b.setSample(0, i, (rnd.nextFloat() * 2 - 1) * std::exp(-i / 500.0f) * (i < 40 ? 1.0f : 0.5f));
            w->writeFromAudioSampleBuffer(b, 0, 2880); } }
        AstroverbAudioProcessor p; p.setPlayConfigDetails(2, 2, 48000, 256); setParam(p, "gain", 0.8f); p.prepareToPlay(48000, 256);
        p.loadIR(ir);
        pump(1500);
        auto r = run(p, 48000, 256, 1.5, 0.3, 110.0);
        printf("with cab IR loaded (%s): peak %.3f FS, rms %.3f FS, CPU %.1f %%, fails %d\n", p.getIRName().toRawUTF8(), r.peak, r.rmsv, r.cpuPct, r.fails);
    }

    // hosted reverb: load a VST3 given on the command line (the test uses the plugin's own VST3 as a stand-in)
    if (argc > 1)
    {
        AstroverbAudioProcessor p; p.setPlayConfigDetails(2, 2, 48000, 256); p.prepareToPlay(48000, 256);
        bool done = false; juce::String err;
        p.loadReverbPlugin(juce::File(argv[1]), [&](const juce::String& e) { err = e; done = true; });
        for (int i = 0; i < 60 && !done; ++i) pump(100);
        printf("hosted plugin: loaded=%d name='%s' error='%s'\n", (int) (p.getReverbInstance() != nullptr), p.getReverbName().toRawUTF8(), err.toRawUTF8());
        if (p.getReverbInstance())
        {
            auto r = run(p, 48000, 256, 1.5, 0.3, 110.0);
            printf("  amp running with hosted plugin in the reverb slot: peak %.3f FS, CPU %.1f %%, fails %d, NaNs %.0f\n", r.peak, r.cpuPct, r.fails, r.nan);
            juce::MemoryBlock mb; p.getStateInformation(mb);
            AstroverbAudioProcessor p2; p2.setPlayConfigDetails(2, 2, 48000, 256); p2.prepareToPlay(48000, 256);
            p2.setStateInformation(mb.getData(), (int) mb.getSize());
            pump(1500);
            printf("  state round-trip: restored reverb plugin = %d, name '%s'\n", (int) (p2.getReverbInstance() != nullptr), p2.getReverbName().toRawUTF8());
            p.clearReverbPlugin();
        }
    }
    return 0;
}
