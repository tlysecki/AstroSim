#pragma once
#include "PluginProcessor.h"

class AstroverbEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit AstroverbEditor(AstroverbAudioProcessor&);
    ~AstroverbEditor() override;
    void paint(juce::Graphics&) override;
    void resized() override;

private:
    struct Knob
    {
        juce::Slider slider{ juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow };
        juce::Label label;
        std::unique_ptr<juce::SliderParameterAttachment> attach;
    };
    void addKnob(Knob& k, const juce::String& paramId, const juce::String& text);
    void timerCallback() override;
    void openReverbWindow();
    void closeReverbWindow();

    AstroverbAudioProcessor& proc;
    Knob gain, bass, mid, treble, volume, presence, reverb, feedback, bias, sag, bright, inTrim, outLevel, cabLevel, sendTrim, retTrim;
    juce::ComboBox toneBox, qualityBox;
    juce::ToggleButton satButton{ "Transformer saturation" }, cabButton{ "Cab IR on" };
    juce::TextButton loadIR{ "Load cab IR..." }, loadReverb{ "Load reverb plugin..." }, showReverb{ "Show" }, removeReverb{ "Remove" };
    juce::Label irName, reverbName, hint;
    std::unique_ptr<juce::ComboBoxParameterAttachment> toneAtt, qualityAtt;
    std::unique_ptr<juce::ButtonParameterAttachment> satAtt, cabAtt;
    std::unique_ptr<juce::FileChooser> chooser;
    std::unique_ptr<juce::DocumentWindow> reverbWindow;
    float meter = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AstroverbEditor)
};
