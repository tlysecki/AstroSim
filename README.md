# Astroverb 16

A circuit-level model of the Soldano Astroverb 16 (preamp, phase inverter, two EL84s, output transformer,
global feedback, speaker load), with a built-in cabinet IR loader and a slot that hosts any VST3 reverb at the
position of the amp's own reverb. Windows VST3 plugin and standalone app.

## Getting the Windows build (no programming needed)
1. Create a free account at github.com and make a **new repository** (private is fine).
2. Upload everything in this folder to it (drag the files into the repository page, or use GitHub Desktop).
   Keep the `.github` folder; it holds the build instructions.
3. Open the **Actions** tab. The build starts by itself (or choose *Build Astroverb 16 (Windows)* -> *Run workflow*).
   The first run takes roughly 10-20 minutes because it compiles JUCE.
4. When it shows a green tick, open the run and download **Astroverb16-VST3** and/or **Astroverb16-Standalone**
   from the *Artifacts* section at the bottom.

## Installing in Reaper
1. Unzip the download. Copy the folder `Astroverb 16.vst3` into `C:\Program Files\Common Files\VST3\`.
2. In Reaper: Options -> Preferences -> Plug-ins -> VST -> *Re-scan*.
3. Put "Astroverb 16" on a track fed by your guitar (DI/clean). Mono or stereo input is fine (stereo is summed).

## Using it
- **Cab**: *Load cab IR* and choose a .wav impulse response. With none loaded you hear the amp's raw speaker output with no cabinet, so load an IR for a realistic sound.
- **Reverb**: *Load reverb plugin*, pick any .vst3 reverb, and set that reverb's own mix to **100% wet** (the amp does the dry/wet blend with its Reverb knob). It runs in 64-sample chunks, so it adds a tiny delay inside the reverb path only.
- **Quality**: *High (2x)* oversamples when the host rate is 50 kHz or lower (more CPU, cleaner top end). *Standard* runs the circuit at the host rate.
- **Tone stack**: *Atomic* (47k slope resistor) or *As drawn* (1k). Bright cap, feedback, bias and sag are knobs.
- Output has a soft safety limiter; turn *Output* down if the meter pins.

## Requirements and notes
- A CPU with **AVX2** (roughly anything since 2013-2015). Expect a good fraction of one core per instance at High.
- The Windows build has not been run on a Windows machine by the author of this code; the first Actions run is the first real compile on MSVC. If it fails, send the log.
- JUCE is used under its licensing terms (AGPLv3 / commercial). Fine for personal use; check before distributing a binary.
