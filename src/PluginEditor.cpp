#include "PluginEditor.h"

namespace
{
class ReverbWindow : public juce::DocumentWindow
{
public:
    ReverbWindow(juce::AudioProcessorEditor* ed, const juce::String& name, std::function<void()> onClose)
        : DocumentWindow(name, juce::Colours::darkgrey, DocumentWindow::closeButton), closeFn(std::move(onClose))
    {
        setUsingNativeTitleBar(true);
        setContentOwned(ed, true);
        setResizable(false, false);
        centreWithSize(getWidth(), getHeight());
        setVisible(true);
    }
    void closeButtonPressed() override { if (closeFn) closeFn(); }
private:
    std::function<void()> closeFn;
};
}

AstroverbEditor::AstroverbEditor(AstroverbAudioProcessor& p) : AudioProcessorEditor(&p), proc(p)
{
    addKnob(gain, "gain", "Gain"); addKnob(bass, "bass", "Bass"); addKnob(mid, "mid", "Mid"); addKnob(treble, "treble", "Treble");
    addKnob(volume, "volume", "Volume"); addKnob(presence, "presence", "Presence"); addKnob(reverb, "reverb", "Reverb");
    addKnob(feedback, "feedback", "Feedback"); addKnob(bias, "bias", "Bias"); addKnob(sag, "sag", "Sag"); addKnob(bright, "bright", "Bright");
    addKnob(inTrim, "intrim", "Input trim"); addKnob(outLevel, "outlevel", "Output"); addKnob(cabLevel, "cablevel", "Cab level");
    addKnob(sendTrim, "sendtrim", "Rev send"); addKnob(retTrim, "rettrim", "Rev return");

    toneBox.addItemList({ "Atomic (47k slope)", "As drawn (1k slope)" }, 1);
    qualityBox.addItemList({ "Standard", "High (2x)" }, 1);
    addAndMakeVisible(toneBox); addAndMakeVisible(qualityBox); addAndMakeVisible(satButton); addAndMakeVisible(cabButton);
    toneAtt = std::make_unique<juce::ComboBoxParameterAttachment>(*proc.apvts.getParameter("tonestack"), toneBox);
    qualityAtt = std::make_unique<juce::ComboBoxParameterAttachment>(*proc.apvts.getParameter("quality"), qualityBox);
    satAtt = std::make_unique<juce::ButtonParameterAttachment>(*proc.apvts.getParameter("otsat"), satButton);
    cabAtt = std::make_unique<juce::ButtonParameterAttachment>(*proc.apvts.getParameter("cabon"), cabButton);

    for (auto* b : { &loadIR, &loadReverb, &showReverb, &removeReverb }) addAndMakeVisible(*b);
    for (auto* l : { &irName, &reverbName, &hint }) { addAndMakeVisible(*l); l->setColour(juce::Label::textColourId, juce::Colours::lightgrey); l->setFont(juce::FontOptions(13.0f)); }
    hint.setText("Reverb plugin: set its mix to 100% wet. Quality: Standard runs the circuit at the host rate, High at 2x.", juce::dontSendNotification);

    loadIR.onClick = [this]
    {
        chooser = std::make_unique<juce::FileChooser>("Choose a cabinet impulse response", juce::File(), "*.wav;*.aif;*.aiff;*.flac");
        chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                             [this](const juce::FileChooser& fc) { if (fc.getResult().existsAsFile()) proc.loadIR(fc.getResult()); });
    };
    loadReverb.onClick = [this]
    {
        chooser = std::make_unique<juce::FileChooser>("Choose a VST3 reverb", juce::File(), "*.vst3");
        chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles | juce::FileBrowserComponent::canSelectDirectories,
                             [this](const juce::FileChooser& fc)
                             {
                                 const auto f = fc.getResult();
                                 if (!f.exists()) return;
                                 closeReverbWindow();
                                 reverbName.setText("Loading...", juce::dontSendNotification);
                                 proc.loadReverbPlugin(f, [this](const juce::String& err)
                                 {
                                     if (err.isNotEmpty()) juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "Reverb plugin", err);
                                 });
                             });
    };
    showReverb.onClick = [this] { openReverbWindow(); };
    removeReverb.onClick = [this] { closeReverbWindow(); proc.clearReverbPlugin(); };

    setSize(900, 560);
    startTimerHz(15);
}

AstroverbEditor::~AstroverbEditor() { closeReverbWindow(); }

void AstroverbEditor::addKnob(Knob& k, const juce::String& id, const juce::String& text)
{
    addAndMakeVisible(k.slider); addAndMakeVisible(k.label);
    k.label.setText(text, juce::dontSendNotification);
    k.label.setJustificationType(juce::Justification::centred);
    k.label.setColour(juce::Label::textColourId, juce::Colour(0xffe8d9b0));
    k.slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 80, 18);
    k.slider.setColour(juce::Slider::rotarySliderFillColourId, juce::Colour(0xffd9a441));
    k.attach = std::make_unique<juce::SliderParameterAttachment>(*proc.apvts.getParameter(id), k.slider);
}

void AstroverbEditor::openReverbWindow()
{
    if (reverbWindow) { reverbWindow->toFront(true); return; }
    if (auto* inst = proc.getReverbInstance())
        if (inst->hasEditor())
            if (auto* ed = inst->createEditorIfNeeded())
                reverbWindow = std::make_unique<ReverbWindow>(ed, inst->getName(), [this] { closeReverbWindow(); });
}

void AstroverbEditor::closeReverbWindow()
{
    if (reverbWindow) { reverbWindow->setVisible(false); reverbWindow.reset(); }
}

void AstroverbEditor::timerCallback()
{
    irName.setText(proc.getIRName().isEmpty() ? "No cab IR loaded (output is the bare amp)" : proc.getIRName(), juce::dontSendNotification);
    reverbName.setText(proc.getReverbName().isEmpty() ? "No reverb loaded" : proc.getReverbName(), juce::dontSendNotification);
    meter = juce::jmax(proc.meterOut.exchange(0.0f), meter * 0.85f);
    repaint(juce::Rectangle<int>(getWidth() - 160, 14, 140, 18));
}

void AstroverbEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff1a1714));
    g.setColour(juce::Colour(0xffd9a441));
    g.setFont(juce::FontOptions(28.0f, juce::Font::bold));
    g.drawText("ASTROVERB 16", 20, 8, 360, 36, juce::Justification::left);
    g.setColour(juce::Colours::grey);
    g.setFont(juce::FontOptions(13.0f));
    g.drawText("circuit model", 22, 40, 200, 16, juce::Justification::left);
    // level meter
    auto r = juce::Rectangle<float>((float) getWidth() - 160.0f, 16.0f, 140.0f, 12.0f);
    g.setColour(juce::Colour(0xff2d2823)); g.fillRoundedRectangle(r, 3.0f);
    const float lv = juce::jlimit(0.0f, 1.0f, meter);
    g.setColour(lv > 0.95f ? juce::Colours::red : juce::Colour(0xffd9a441));
    g.fillRoundedRectangle(r.withWidth(r.getWidth() * lv), 3.0f);
}

void AstroverbEditor::resized()
{
    auto place = [](std::vector<Knob*> row, juce::Rectangle<int> area)
    {
        const int w = area.getWidth() / (int) row.size();
        for (auto* k : row)
        {
            auto cell = area.removeFromLeft(w);
            k->label.setBounds(cell.removeFromTop(18));
            k->slider.setBounds(cell.reduced(6, 0));
        }
    };
    auto area = getLocalBounds().reduced(20);
    area.removeFromTop(56);
    place({ &gain, &bass, &mid, &treble, &volume, &presence, &reverb }, area.removeFromTop(120));
    area.removeFromTop(8);
    place({ &feedback, &bias, &sag, &bright, &inTrim, &outLevel, &cabLevel }, area.removeFromTop(120));
    area.removeFromTop(10);

    auto row = area.removeFromTop(28);
    toneBox.setBounds(row.removeFromLeft(200)); row.removeFromLeft(12);
    qualityBox.setBounds(row.removeFromLeft(140)); row.removeFromLeft(12);
    satButton.setBounds(row.removeFromLeft(190)); cabButton.setBounds(row.removeFromLeft(120));
    area.removeFromTop(10);
    row = area.removeFromTop(28);
    loadIR.setBounds(row.removeFromLeft(150)); row.removeFromLeft(10); irName.setBounds(row);
    area.removeFromTop(8);
    row = area.removeFromTop(28);
    loadReverb.setBounds(row.removeFromLeft(170)); row.removeFromLeft(8);
    showReverb.setBounds(row.removeFromLeft(60)); row.removeFromLeft(6);
    removeReverb.setBounds(row.removeFromLeft(70)); row.removeFromLeft(10);
    reverbName.setBounds(row.removeFromLeft(260)); row.removeFromLeft(10);
    {
        auto k1 = row.removeFromLeft(row.getWidth() / 2);
        sendTrim.label.setBounds(k1.removeFromLeft(60)); sendTrim.slider.setSliderStyle(juce::Slider::LinearHorizontal);
        sendTrim.slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 18); sendTrim.slider.setBounds(k1);
        retTrim.label.setBounds(row.removeFromLeft(60)); retTrim.slider.setSliderStyle(juce::Slider::LinearHorizontal);
        retTrim.slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 18); retTrim.slider.setBounds(row);
    }
    area.removeFromTop(10);
    hint.setBounds(area.removeFromTop(22));
}
