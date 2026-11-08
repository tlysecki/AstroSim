#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>
#include <cstring>

namespace
{
constexpr double kInputVoltsPerFS = 1.0;    // 0 dBFS at the input = 1 V peak at the guitar jack (before the input trim)
constexpr double kOutScale = 1.0 / 25.0;     // speaker volts -> full scale (a hard-clipped square at ~20 V rms sits near -2 dBFS)
constexpr double kSendVoltsPerFS = 40.0;     // V2a plate swing that maps to full scale at the reverb send
constexpr double kReturnVoltsPerFS = 0.113 * kSendVoltsPerFS;   // reverb return level at which wet ~ dry at max knob

inline double dbToLin(double db) { return std::pow(10.0, db / 20.0); }

inline float softLimit(float x)              // safety only: transparent below 0.8, never exceeds 1.0
{
    const float a = std::fabs(x);
    if (a <= 0.8f) return x;
    const float y = 0.8f + 0.2f * std::tanh((a - 0.8f) * 5.0f);
    return x < 0.0f ? -y : y;
}
}

double AstroverbAudioProcessor::feedbackToOhms(double x) { return 4.7e3 * std::pow(100e3 / 4.7e3, 1.0 - x); }
double AstroverbAudioProcessor::brightToFarads(double x) { return 1500e-12 * x * x; }

juce::AudioProcessorValueTreeState::ParameterLayout AstroverbAudioProcessor::createLayout()
{
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout l;
    auto pct = [](float v, int) { return String(v * 10.0f, 1); };      // knobs read 0..10 like the amp
    auto knob = [&](const char* id, const char* name, float def)
    {
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{ id, 1 }, name, NormalisableRange<float>(0.0f, 1.0f, 0.001f), def,
              AudioParameterFloatAttributes().withStringFromValueFunction(pct)));
    };
    knob("gain", "Gain", 0.65f);
    knob("bass", "Bass", 0.5f);
    knob("mid", "Mid", 0.5f);
    knob("treble", "Treble", 0.5f);
    knob("volume", "Volume", 0.7f);
    knob("presence", "Presence", 0.5f);
    knob("reverb", "Reverb", 0.4f);
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{ "feedback", 1 }, "Feedback", NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.7535f,
          AudioParameterFloatAttributes().withStringFromValueFunction([](float v, int) { return String(feedbackToOhms(v) / 1000.0, 1) + "k"; })));
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{ "bias", 1 }, "Bias", NormalisableRange<float>(-24.0f, -8.0f, 0.1f), -16.0f,
          AudioParameterFloatAttributes().withStringFromValueFunction([](float v, int) { return String(v, 1) + " V"; })));
    knob("sag", "Sag", 0.75f);
    l.add(std::make_unique<AudioParameterFloat>(ParameterID{ "bright", 1 }, "Bright", NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.56f,
          AudioParameterFloatAttributes().withStringFromValueFunction([](float v, int) { return String(brightToFarads(v) * 1e12, 0) + " pF"; })));
    l.add(std::make_unique<AudioParameterChoice>(ParameterID{ "tonestack", 1 }, "Tone stack", StringArray{ "Atomic (47k slope)", "As drawn (1k slope)" }, 0));
    l.add(std::make_unique<AudioParameterBool>(ParameterID{ "otsat", 1 }, "Transformer sat", true));
    auto dbParam = [&](const char* id, const char* name, float lo, float hi, float def)
    {
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{ id, 1 }, name, NormalisableRange<float>(lo, hi, 0.1f), def,
              AudioParameterFloatAttributes().withStringFromValueFunction([](float v, int) { return String(v, 1) + " dB"; })));
    };
    dbParam("intrim", "Input trim", -24.0f, 24.0f, 0.0f);
    dbParam("outlevel", "Output", -30.0f, 12.0f, 0.0f);
    dbParam("sendtrim", "Reverb send", -24.0f, 24.0f, 0.0f);
    dbParam("rettrim", "Reverb return", -24.0f, 24.0f, 0.0f);
    dbParam("cablevel", "Cab level", -24.0f, 12.0f, 0.0f);
    l.add(std::make_unique<AudioParameterBool>(ParameterID{ "cabon", 1 }, "Cab IR", true));
    l.add(std::make_unique<AudioParameterChoice>(ParameterID{ "quality", 1 }, "Quality", StringArray{ "Standard", "High (2x)" }, 1));
    return l;
}

AstroverbAudioProcessor::AstroverbAudioProcessor()
    : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                      .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "STATE", createLayout())
{
    pGain = apvts.getRawParameterValue("gain"); pBass = apvts.getRawParameterValue("bass"); pMid = apvts.getRawParameterValue("mid");
    pTreble = apvts.getRawParameterValue("treble"); pVol = apvts.getRawParameterValue("volume"); pPres = apvts.getRawParameterValue("presence");
    pFb = apvts.getRawParameterValue("feedback"); pBias = apvts.getRawParameterValue("bias"); pSag = apvts.getRawParameterValue("sag");
    pBright = apvts.getRawParameterValue("bright"); pTone = apvts.getRawParameterValue("tonestack"); pSat = apvts.getRawParameterValue("otsat");
    pIn = apvts.getRawParameterValue("intrim"); pOut = apvts.getRawParameterValue("outlevel"); pQuality = apvts.getRawParameterValue("quality");
    pRev = apvts.getRawParameterValue("reverb"); pSendTrim = apvts.getRawParameterValue("sendtrim"); pRetTrim = apvts.getRawParameterValue("rettrim");
    pCabOn = apvts.getRawParameterValue("cabon"); pCabLevel = apvts.getRawParameterValue("cablevel");
    formatManager.addFormat(new juce::VST3PluginFormat());
}

AstroverbAudioProcessor::~AstroverbAudioProcessor() { clearReverbPlugin(); }

bool AstroverbAudioProcessor::isBusesLayoutSupported(const BusesLayout& l) const
{
    const auto in = l.getMainInputChannelSet(), out = l.getMainOutputChannelSet();
    if (in != juce::AudioChannelSet::mono() && in != juce::AudioChannelSet::stereo()) return false;
    return out == juce::AudioChannelSet::stereo() || out == juce::AudioChannelSet::mono();
}

void AstroverbAudioProcessor::buildParams(astro::Params& p)
{
    p.gain = sGain.getCurrentValue(); p.bass = sBass.getCurrentValue(); p.mid = sMid.getCurrentValue();
    p.treble = sTreble.getCurrentValue(); p.volume = sVol.getCurrentValue(); p.presence = sPres.getCurrentValue();
    p.fbR = feedbackToOhms(sFb.getCurrentValue()); p.bias = sBias.getCurrentValue(); p.sag = sSag.getCurrentValue();
    p.brightC = std::max(1e-12, brightToFarads(sBright.getCurrentValue()));
    p.slopeR = ((int) pTone->load() == 0) ? 47e3 : 1e3;
    p.revLevel = sRev.getCurrentValue();
    p.otSat = pSat->load() > 0.5f;
    p.speakerLoad = true;
}

void AstroverbAudioProcessor::reconfigure(int quality)
{
    activeQuality = quality;
    solverFactor = (quality == 1 && hostRate <= 50000.0) ? 2 : 1;
    const double solverRate = hostRate * solverFactor;

    os.reset();
    int latency = 0;
    if (solverFactor == 2)
    {
        os = std::make_unique<juce::dsp::Oversampling<float>>(1, 1, juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR, true, false);
        os->initProcessing((size_t) blockSize);
        latency = (int) std::lround(os->getLatencyInSamples());
    }
    setLatencySamples(latency);
    upBuf.assign((size_t) blockSize * (size_t) solverFactor + 8, 0.0f);

    auto setS = [&](juce::SmoothedValue<double>& s, double v) { s.reset(solverRate, 0.03); s.setCurrentAndTargetValue(v); };
    setS(sGain, *pGain); setS(sBass, *pBass); setS(sMid, *pMid); setS(sTreble, *pTreble); setS(sVol, *pVol);
    setS(sPres, *pPres); setS(sFb, *pFb); setS(sBias, *pBias); setS(sSag, *pSag); setS(sBright, *pBright); setS(sRev, *pRev);
    astro::Params p; buildParams(p);
    core.prepare(solverRate, p);
    lastParams = p; paramsInit = true;
}

void AstroverbAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    hostRate = sampleRate; blockSize = std::max(samplesPerBlock, 64);
    auto setF = [&](juce::SmoothedValue<float>& s, float v) { s.reset(sampleRate, 0.03); s.setCurrentAndTargetValue(v); };
    setF(sInTrim, (float) dbToLin(*pIn)); setF(sOutLevel, (float) dbToLin(*pOut)); setF(sCabLevel, (float) dbToLin(*pCabLevel));
    setF(sSendTrim, (float) dbToLin(*pSendTrim)); setF(sRetTrim, (float) dbToLin(*pRetTrim));

    work.setSize(2, blockSize, false, true, false);
    retHold.assign((size_t) blockSize + 8, 0.0f);
    reconfigure((int) pQuality->load());

    juce::dsp::ProcessSpec spec{ sampleRate, (juce::uint32) blockSize, 2 };
    cab.prepare(spec);
    auto hp = juce::dsp::IIR::Coefficients<float>::makeHighPass(sampleRate, 25.0f, 0.7071f);
    for (int c = 0; c < 2; ++c) { hpf1[c].coefficients = hp; hpf1[c].reset(); hpf2[c].coefficients = hp; hpf2[c].reset(); }

    sendChunk.assign(reverbChunk, 0.0f); wetPlay.assign(reverbChunk, 0.0f);
    sendFill = 0; wetPos = reverbChunk; sendDcState = 0.0;
    reverbBuf.setSize(2, reverbChunk);
    {
        const juce::SpinLock::ScopedLockType sl(reverbLock);
        if (reverb) { reverb->setPlayConfigDetails(2, 2, sampleRate, reverbChunk); reverb->prepareToPlay(sampleRate, reverbChunk); }
    }
}

void AstroverbAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int numIn = getTotalNumInputChannels(), numOut = getTotalNumOutputChannels();
    const int total = buffer.getNumSamples();
    for (int c = numIn; c < numOut; ++c) buffer.clear(c, 0, total);

    // targets from the host-facing parameters
    const double sr = hostRate * solverFactor; juce::ignoreUnused(sr);
    sGain.setTargetValue(*pGain); sBass.setTargetValue(*pBass); sMid.setTargetValue(*pMid); sTreble.setTargetValue(*pTreble);
    sVol.setTargetValue(*pVol); sPres.setTargetValue(*pPres); sFb.setTargetValue(*pFb); sBias.setTargetValue(*pBias);
    sSag.setTargetValue(*pSag); sBright.setTargetValue(*pBright); sRev.setTargetValue(*pRev);
    sInTrim.setTargetValue((float) dbToLin(*pIn)); sOutLevel.setTargetValue((float) dbToLin(*pOut)); sCabLevel.setTargetValue((float) dbToLin(*pCabLevel));
    sSendTrim.setTargetValue((float) dbToLin(*pSendTrim)); sRetTrim.setTargetValue((float) dbToLin(*pRetTrim));
    if ((int) pQuality->load() != activeQuality) reconfigure((int) pQuality->load());

    const bool cabOn = pCabOn->load() > 0.5f && irLoaded.load() && cab.getCurrentIRSize() > 1;
    float peak = 0.0f;

    for (int start = 0; start < total; start += blockSize)
    {
        const int n = std::min(blockSize, total - start);
        float* mono = work.getWritePointer(0);
        const float* inL = buffer.getReadPointer(0, start);
        const float* inR = numIn > 1 ? buffer.getReadPointer(1, start) : nullptr;
        for (int i = 0; i < n; ++i)
        {
            const float x = inR ? 0.5f * (inL[i] + inR[i]) : inL[i];
            mono[i] = x * sInTrim.getNextValue() * (float) kInputVoltsPerFS;
        }

        // ---- up-sample to the solver rate ----
        float* solverIn = mono; int m = n;
        if (solverFactor == 2)
        {
            juce::dsp::AudioBlock<float> blk(&mono, 1, (size_t) n);
            auto up = os->processSamplesUp(blk);
            solverIn = up.getChannelPointer(0); m = (int) up.getNumSamples();
        }

        // ---- circuit, with the reverb loop closed at the amp's own insertion point ----
        float* solverOut = solverFactor == 2 ? solverIn : mono;      // in place
        int ctr = 0;
        for (int i = 0; i < n; ++i)
        {
            if (wetPos >= reverbChunk) wetPos = reverbChunk - 1;
            const double ret = (double) wetPlay[(size_t) wetPos++] * (double) sRetTrim.getNextValue() * kReturnVoltsPerFS;
            double sendAcc = 0.0;
            for (int k = 0; k < solverFactor; ++k)
            {
                const int j = i * solverFactor + k;
                if (ctr == 0)
                {
                    sGain.skip(controlInterval); sBass.skip(controlInterval); sMid.skip(controlInterval); sTreble.skip(controlInterval);
                    sVol.skip(controlInterval); sPres.skip(controlInterval); sFb.skip(controlInterval); sBias.skip(controlInterval);
                    sSag.skip(controlInterval); sBright.skip(controlInterval); sRev.skip(controlInterval);
                    astro::Params p; buildParams(p);
                    if (p.gain != lastParams.gain || p.bass != lastParams.bass || p.mid != lastParams.mid || p.treble != lastParams.treble
                        || p.volume != lastParams.volume || p.presence != lastParams.presence || p.fbR != lastParams.fbR || p.bias != lastParams.bias
                        || p.sag != lastParams.sag || p.brightC != lastParams.brightC || p.slopeR != lastParams.slopeR
                        || p.revLevel != lastParams.revLevel || p.otSat != lastParams.otSat)
                    { core.setParams(p); lastParams = p; }
                }
                ctr = (ctr + 1) % controlInterval;
                solverOut[j] = (float) core.process((double) solverIn[j], ret);
                sendAcc += core.sendVolts();
            }
            // reverb send: V2a plate, DC removed, to full-scale units
            sendAcc /= solverFactor;
            sendDcState += 0.0005 * (sendAcc - sendDcState) * (48000.0 / hostRate);
            sendChunk[(size_t) sendFill++] = (float) ((sendAcc - sendDcState) / kSendVoltsPerFS) * sSendTrim.getNextValue();
            if (sendFill >= reverbChunk) closeReverbLoop(reverbChunk);
        }
        solverFails.store((int) core.failCount(), std::memory_order_relaxed);

        // ---- back to the host rate ----
        if (solverFactor == 2)
        {
            juce::dsp::AudioBlock<float> outBlk(&mono, 1, (size_t) n);
            os->processSamplesDown(outBlk);
        }

        // ---- speaker volts -> signal level, output trim, low cut, safety against NaN ----
        for (int i = 0; i < n; ++i)
        {
            const float og = sOutLevel.getNextValue();
            float y = mono[i] * (float) kOutScale * og;
            if (!std::isfinite(y)) y = 0.0f;
            y = hpf2[0].processSample(hpf1[0].processSample(y));
            mono[i] = y;
        }

        // ---- cab IR (dual mono in, so a stereo IR gives a stereo image) ----
        float* l = work.getWritePointer(0);
        float* r = work.getWritePointer(1);
        if (cabOn)
        {
            std::memcpy(r, l, sizeof(float) * (size_t) n);
            juce::dsp::AudioBlock<float> blk(work.getArrayOfWritePointers(), 2, (size_t) n);
            juce::dsp::ProcessContextReplacing<float> ctx(blk);
            cab.process(ctx);
            for (int i = 0; i < n; ++i) { const float g = sCabLevel.getNextValue(); l[i] *= g; r[i] *= g; }
        }
        else
        {
            std::memcpy(r, l, sizeof(float) * (size_t) n);
            for (int i = 0; i < n; ++i) sCabLevel.getNextValue();
        }

        for (int i = 0; i < n; ++i)
        {
            l[i] = softLimit(l[i]); r[i] = softLimit(r[i]);
            peak = std::max(peak, std::max(std::fabs(l[i]), std::fabs(r[i])));
        }
        std::memcpy(buffer.getWritePointer(0, start), l, sizeof(float) * (size_t) n);
        if (numOut > 1) std::memcpy(buffer.getWritePointer(1, start), r, sizeof(float) * (size_t) n);
    }
    meterOut.store(peak);
}

void AstroverbAudioProcessor::closeReverbLoop(int)
{
    // one full chunk of send is collected: run the hosted reverb on it; the result plays back starting with the next sample
    bool did = false;
    {
        const juce::SpinLock::ScopedTryLockType sl(reverbLock);
        if (sl.isLocked() && reverb != nullptr)
        {
            for (int c = 0; c < 2; ++c) std::memcpy(reverbBuf.getWritePointer(c), sendChunk.data(), sizeof(float) * (size_t) reverbChunk);
            reverb->processBlock(reverbBuf, emptyMidi);
            const float* a = reverbBuf.getReadPointer(0);
            const float* b = reverbBuf.getReadPointer(reverbBuf.getNumChannels() > 1 ? 1 : 0);
            for (int k = 0; k < reverbChunk; ++k) { const float w = 0.5f * (a[k] + b[k]); wetPlay[(size_t) k] = std::isfinite(w) ? w : 0.0f; }
            did = true;
        }
    }
    if (!did) std::fill(wetPlay.begin(), wetPlay.end(), 0.0f);
    sendFill = 0; wetPos = 0;
}

// ------------------------------------------------------------------ IR
void AstroverbAudioProcessor::loadIR(const juce::File& f)
{
    // read the file here so a failure is reported instead of silently leaving the placeholder IR in place
    if (!f.existsAsFile()) { irStatus = "IR file not found: " + f.getFileName(); return; }
    juce::AudioFormatManager fm; fm.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(f));
    if (reader == nullptr || reader->lengthInSamples < 2 || reader->numChannels < 1)
    { irStatus = "Could not read " + f.getFileName() + " (use a PCM .wav, 16/24/32-bit)"; return; }

    const int len = (int) std::min<juce::int64>(reader->lengthInSamples, (juce::int64) (reader->sampleRate * 2.0));   // 2 s is plenty for a cab
    juce::AudioBuffer<float> buf((int) std::min<unsigned int>(reader->numChannels, 2u), len);
    reader->read(&buf, 0, len, 0, true, buf.getNumChannels() > 1);
    const double sr = reader->sampleRate;
    const int chans = buf.getNumChannels();

    irFile = f;
    cab.loadImpulseResponse(std::move(buf), sr, chans > 1 ? juce::dsp::Convolution::Stereo::yes : juce::dsp::Convolution::Stereo::no,
                            juce::dsp::Convolution::Trim::yes, juce::dsp::Convolution::Normalise::yes);
    irStatus = f.getFileName() + "  (" + juce::String(chans == 1 ? "mono" : "stereo") + ", " + juce::String(len * 1000.0 / sr, 0) + " ms, " + juce::String(sr / 1000.0, 1) + " kHz)";
    irLoaded.store(true);
}

// ------------------------------------------------------------------ hosted reverb
juce::String AstroverbAudioProcessor::getReverbName() const { return reverb ? reverb->getName() : juce::String(); }

void AstroverbAudioProcessor::loadReverbPlugin(const juce::File& file, std::function<void(const juce::String&)> done)
{
    juce::OwnedArray<juce::PluginDescription> descs;
    for (auto* fmt : formatManager.getFormats()) fmt->findAllTypesForFile(descs, file.getFullPathName());
    if (descs.isEmpty()) { if (done) done("That file is not a VST3 plugin."); return; }
    juce::PluginDescription d = *descs[0];
    for (auto* x : descs) if (!x->isInstrument) { d = *x; break; }
    const double sr = hostRate; const int bs = reverbChunk;
    formatManager.createPluginInstanceAsync(d, sr, bs, [this, file, done](std::unique_ptr<juce::AudioPluginInstance> inst, const juce::String& err)
    {
        if (!inst) { if (done) done(err.isEmpty() ? "Could not load the plugin." : err); return; }
        inst->enableAllBuses();
        juce::AudioProcessor::BusesLayout lay = inst->getBusesLayout();
        if (inst->getBusCount(true) > 0)  lay.getChannelSet(true, 0)  = juce::AudioChannelSet::stereo();
        if (inst->getBusCount(false) > 0) lay.getChannelSet(false, 0) = juce::AudioChannelSet::stereo();
        inst->setBusesLayout(lay);
        inst->setPlayConfigDetails(2, 2, hostRate, reverbChunk);
        inst->prepareToPlay(hostRate, reverbChunk);
        if (pendingReverbState.getSize() > 0) { inst->setStateInformation(pendingReverbState.getData(), (int) pendingReverbState.getSize()); pendingReverbState.reset(); }
        std::unique_ptr<juce::AudioPluginInstance> old;
        {
            const juce::SpinLock::ScopedLockType sl(reverbLock);
            old = std::move(reverb); reverb = std::move(inst); reverbFile = file;
        }
        old.reset();
        if (done) done({});
    });
}

void AstroverbAudioProcessor::clearReverbPlugin()
{
    std::unique_ptr<juce::AudioPluginInstance> old;
    { const juce::SpinLock::ScopedLockType sl(reverbLock); old = std::move(reverb); reverbFile = juce::File(); }
    old.reset();
}

// ------------------------------------------------------------------ state
void AstroverbAudioProcessor::getStateInformation(juce::MemoryBlock& dest)
{
    auto state = apvts.copyState();
    state.setProperty("irPath", irFile.getFullPathName(), nullptr);
    state.setProperty("reverbPath", reverbFile.getFullPathName(), nullptr);
    if (reverb)
    {
        juce::MemoryBlock mb; reverb->getStateInformation(mb);
        state.setProperty("reverbState", mb.toBase64Encoding(), nullptr);
    }
    if (auto xml = state.createXml()) copyXmlToBinary(*xml, dest);
}

void AstroverbAudioProcessor::setStateInformation(const void* data, int size)
{
    if (auto xml = getXmlFromBinary(data, size))
    {
        auto st = juce::ValueTree::fromXml(*xml);
        if (!st.isValid()) return;
        apvts.replaceState(st);
        const juce::File ir(st.getProperty("irPath").toString());
        if (ir.existsAsFile()) loadIR(ir);
        const juce::File rv(st.getProperty("reverbPath").toString());
        if (rv.exists())
        {
            juce::MemoryBlock mb; mb.fromBase64Encoding(st.getProperty("reverbState").toString());
            pendingReverbState = mb;
            juce::MessageManager::callAsync([this, rv] { loadReverbPlugin(rv); });
        }
    }
}

juce::AudioProcessorEditor* AstroverbAudioProcessor::createEditor() { return new AstroverbEditor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new AstroverbAudioProcessor(); }
