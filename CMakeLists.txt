cmake_minimum_required(VERSION 3.22)
project(Astroverb VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(JUCE
  GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
  GIT_TAG 8.0.8
  GIT_SHALLOW ON)
FetchContent_MakeAvailable(JUCE)

# ---- the circuit solver (no JUCE dependency) ----
add_library(astroverb_core STATIC src/AstroverbCore.cpp src/AstroverbCore.h)
target_include_directories(astroverb_core PUBLIC src)
if(MSVC)
  target_compile_options(astroverb_core PRIVATE /arch:AVX2 /O2 /fp:precise)
else()
  target_compile_options(astroverb_core PRIVATE -O3 -mavx2 -mfma)
endif()
set_target_properties(astroverb_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- the plugin ----
juce_add_plugin(Astroverb
  COMPANY_NAME "Tim Lysecki"
  BUNDLE_ID com.timlysecki.astroverb16
  PLUGIN_MANUFACTURER_CODE Tlys
  PLUGIN_CODE Asv1
  FORMATS VST3 Standalone
  PRODUCT_NAME "Astroverb 16"
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  VST3_CATEGORIES Fx
  COPY_PLUGIN_AFTER_BUILD FALSE)

juce_generate_juce_header(Astroverb)
target_sources(Astroverb PRIVATE
  src/PluginProcessor.cpp src/PluginProcessor.h
  src/PluginEditor.cpp src/PluginEditor.h)
target_compile_definitions(Astroverb PUBLIC
  JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0
  JUCE_PLUGINHOST_VST3=1 JUCE_STRICT_REFCOUNTEDPOINTER=1
  JUCE_DISPLAY_SPLASH_SCREEN=0)
target_link_libraries(Astroverb
  PRIVATE astroverb_core juce::juce_audio_utils juce::juce_dsp
  PUBLIC juce::juce_recommended_config_flags)

# ---- offline test programs (not part of the plugin) ----
option(ASTROVERB_BUILD_TESTS "Build the offline test programs" OFF)
if(ASTROVERB_BUILD_TESTS)
  add_executable(core_test tests/core_test.cpp)
  target_link_libraries(core_test PRIVATE astroverb_core)
  juce_add_console_app(processor_test PRODUCT_NAME "processor_test")
  target_sources(processor_test PRIVATE tests/ProcessorTest.cpp src/PluginProcessor.cpp src/PluginEditor.cpp)
  target_include_directories(processor_test PRIVATE src)
  target_compile_definitions(processor_test PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_PLUGINHOST_VST3=1 JUCE_MODAL_LOOPS_PERMITTED=1)
  target_link_libraries(processor_test PRIVATE astroverb_core juce::juce_audio_utils juce::juce_dsp juce::juce_recommended_config_flags)
endif()
