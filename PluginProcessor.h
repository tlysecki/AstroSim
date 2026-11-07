#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include "AstroverbCore.h"

// Astroverb 16: circuit-level model of the Soldano Astroverb 16 (preamp, phase inverter, 2xEL84, output
// transformer, global feedback, speaker load) + built-in cab IR + a slot that hosts any VST3 reverb at the
// position of the amp's own reverb (driven from V2a's plate, returned into the V2b/V3a grid).
class AstroverbAudioProcessor : public juce::AudioProcessor
{
public:
    AstroverbAudioProcessor();
    ~AstroverbAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using juce::AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Astroverb 16"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 2.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // ---- cab IR ----
    void loadIR(const juce::File& f);
    juce::String getIRName() const { return irFile.existsAsFile() ? irFile.getFileName() : juce::String(); }

    // ---- hosted reverb (any VST3) ----
    void loadReverbPlugin(const juce::File& vst3, std::function<void(const juce::String& error)> done = {});
    void clearReverbPlugin();
    juce::String getReverbName() const;
    juce::AudioPluginInstance* getReverbInstance() const { return reverb.get(); }

    // ---- state shared with the editor ----
    std::atomic<float> meterOut{0.0f};          // output peak (0..1+) for a simple level meter
    std::atomic<int> solverFails{0};

    juce::AudioProcessorValueTreeState apvts;
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

    // mapping helpers (also used by the editor/tests)
    static double feedbackToOhms(double x);
    static double brightToFarads(double x);

private:
    struct Smoothed { juce::SmoothedValue<double> v; };

    void reconfigure(int quality);
    void buildParams(astro::Params& p);
    void closeReverbLoop(int numHostSamples);

    // circuit
    astro::Core core;
    astro::Params lastParams;
    bool paramsInit = false;
    int solverFactor = 2;
    int activeQuality = -1;
    double hostRate = 48000.0;
    int blockSize = 512;
    std::unique_ptr<juce::dsp::Oversampling<float>> os;
    juce::SmoothedValue<double> sGain, sBass, sMid, sTreble, sVol, sPres, sFb, sBias, sSag, sBright, sRev;
    juce::SmoothedValue<float> sInTrim, sOutLevel, sCabLevel, sSendTrim, sRetTrim;
    static constexpr int controlInterval = 32;        // solver samples between matrix updates

    // post chain
    juce::dsp::Convolution cab { juce::dsp::Convolution::NonUniform { 256 } };
    juce::File irFile;
    std::atomic<bool> irLoaded{false};      // JUCE's convolution holds a 1-sample placeholder IR until a real one is loaded
    juce::dsp::IIR::Filter<float> hpf1[2], hpf2[2];
    juce::AudioBuffer<float> work;                   // mono float scratch
    std::vector<float> upBuf, retHold;

    // reverb host
    static constexpr int reverbChunk = 64;
    juce::SpinLock reverbLock;
    std::unique_ptr<juce::AudioPluginInstance> reverb;
    juce::File reverbFile;
    juce::MemoryBlock pendingReverbState;
    juce::AudioBuffer<float> reverbBuf;
    std::vector<float> sendChunk, wetPlay;
    int sendFill = 0, wetPos = reverbChunk;
    double sendDc = 0.0, sendDcState = 0.0;
    juce::MidiBuffer emptyMidi;
    juce::AudioPluginFormatManager formatManager;

    std::atomic<float>* pGain; std::atomic<float>* pBass; std::atomic<float>* pMid; std::atomic<float>* pTreble;
    std::atomic<float>* pVol; std::atomic<float>* pPres; std::atomic<float>* pFb; std::atomic<float>* pBias;
    std::atomic<float>* pSag; std::atomic<float>* pBright; std::atomic<float>* pTone; std::atomic<float>* pSat;
    std::atomic<float>* pIn; std::atomic<float>* pOut; std::atomic<float>* pQuality; std::atomic<float>* pRev;
    std::atomic<float>* pSendTrim; std::atomic<float>* pRetTrim; std::atomic<float>* pCabOn; std::atomic<float>* pCabLevel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AstroverbAudioProcessor)
};
