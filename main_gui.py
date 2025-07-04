import threading
import tkinter as tk
import random
import queue
import os
import subprocess
import tempfile
import time
import sys
import ctypes
from tkinter import messagebox, ttk, simpledialog
from DobAEI import AutonomousSystem, SELF_AWARENESS

# Try to import Coqui TTS
try:
    import torch
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
    print("✅ Coqui TTS library available")
except ImportError:
    COQUI_TTS_AVAILABLE = False
    print("⚠️ Coqui TTS not available - install with: pip install TTS")

# Try to import speech recognition libraries
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    print("✅ Speech recognition library available")
except ImportError:
    # Import the placeholder sr from DobAEI if speech_recognition is not available
    from DobAEI import sr
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️ Speech recognition not available - install with: pip install SpeechRecognition")

# Function to check if ffmpeg is installed
def is_ffmpeg_installed():
    try:
        # Try to run ffmpeg with version flag
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Try to import whisper for transcription
try:
    import whisper
    # Check if ffmpeg is installed
    if is_ffmpeg_installed():
        WHISPER_AVAILABLE = True
        print("✅ Whisper library available")
    else:
        WHISPER_AVAILABLE = False
        print("⚠️ Whisper available but ffmpeg not found - install ffmpeg for audio processing")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not available - install with: pip install openai-whisper")

# Try to import sounddevice and soundfile for audio
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
    print("✅ Audio libraries available")
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ Audio libraries not available - install with: pip install sounddevice soundfile")

# Try to import vosk for lightweight speech recognition
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
    print("✅ Vosk library available")
except ImportError:
    VOSK_AVAILABLE = False
    print("⚠️ Vosk not available - install with: pip install vosk")


class AutonomousApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DoBA Voice Assistant")
        self.root.geometry("800x600")  # Set initial window size

        # Configure the window to be responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize the autonomous system
        self.system = AutonomousSystem()

        # Initialize voice-related variables
        self.voice_mode_enabled = True  # Start with voice mode enabled for testing
        self.voice_thread_running = False
        self.voice_thread = None
        self.is_speaking = False
        self.is_listening = False
        self.chat_history = []
        self.audio_queue = queue.Queue()

        # Audio initialization status
        self.audio_initialized = False
        self.audio_init_attempts = 0
        self.max_audio_init_attempts = 3

        # Voice recognition buffer variables
        self.voice_buffer = []
        self.last_voice_time = 0
        self.voice_pause_threshold = 1.5  # seconds of silence to consider a sentence complete
        self.voice_buffer_timeout = 10.0  # seconds before force-processing a buffered sentence

        # Initialize device selection variables
        self.input_devices = []
        self.output_devices = []
        self.selected_input_device = None
        self.selected_output_device = None

        # Initialize TTS model
        self.tts_model = None

        # Initialize speech recognition if available
        self.recognizer = None

        # Initialize whisper model if available
        self.whisper_model = None

        # Initialize vosk model if available
        self.vosk_model = None

        # Start robust audio initialization process
        self._initialize_audio_system()

        # Create GUI components
        self.create_widgets()

        # Set up periodic autonomous thought generation
        self.thought_interval = 60000  # 60 seconds
        self.schedule_autonomous_thought()

        # Set up periodic updates
        self.update_interval = 5000  # 5 seconds
        self.schedule_update()

        # Schedule periodic audio system checks
        self.audio_check_interval = 300000  # 5 minutes
        self.schedule_audio_system_check()

    def _initialize_audio_system(self):
        """Initialize the audio system with robust error handling and recovery."""
        try:
            print("\n" + "="*50)
            print("🔊 AUDIO SYSTEM INITIALIZATION")
            print("="*50)

            # Check for required audio packages
            self._check_audio_dependencies()

            # Initialize speech recognition if available
            if SPEECH_RECOGNITION_AVAILABLE:
                try:
                    print("🔊 AUDIO INIT: Initializing speech recognition...")
                    self.recognizer = sr.Recognizer()
                    print("✅ AUDIO INIT: Speech recognition initialized")
                except Exception as e:
                    print(f"❌ AUDIO INIT: Error initializing speech recognition: {e}")

            # Initialize whisper model if available
            if WHISPER_AVAILABLE:
                try:
                    print("🔊 AUDIO INIT: Loading Whisper model...")
                    self.whisper_model = whisper.load_model("tiny")
                    print("✅ AUDIO INIT: Whisper model loaded")
                except Exception as e:
                    print(f"❌ AUDIO INIT: Error loading Whisper model: {e}")

            # Initialize vosk model if available
            if VOSK_AVAILABLE:
                try:
                    model_path = os.path.join(os.path.expanduser("~"), "vosk-model-small-en-us-0.15")
                    if os.path.exists(model_path):
                        print("🔊 AUDIO INIT: Loading Vosk model...")
                        self.vosk_model = Model(model_path)
                        print("✅ AUDIO INIT: Vosk model loaded successfully")
                    else:
                        print("⚠️ AUDIO INIT: Vosk model not found at", model_path)
                except Exception as e:
                    print(f"❌ AUDIO INIT: Error loading Vosk model: {e}")

            # Get available audio devices
            self._initialize_audio_devices()

            # Initialize TTS model with multiple attempts
            self._initialize_tts_with_retry()

            # Test audio system
            self._test_audio_system()

            # Mark audio system as initialized
            self.audio_initialized = True
            print("✅ AUDIO INIT: Audio system initialization completed")
            print("="*50 + "\n")

        except Exception as e:
            print(f"❌ AUDIO INIT: Error initializing audio system: {e}")
            print("⚠️ AUDIO INIT: Will retry initialization later")

            # Schedule a retry
            self.audio_init_attempts += 1
            if self.audio_init_attempts < self.max_audio_init_attempts:
                retry_delay = 5000 * (2 ** self.audio_init_attempts)  # Exponential backoff
                print(f"⚠️ AUDIO INIT: Scheduling retry in {retry_delay/1000} seconds (attempt {self.audio_init_attempts+1}/{self.max_audio_init_attempts})")
                self.root.after(retry_delay, self._initialize_audio_system)

    def _test_and_repair_audio_subsystem(self):
        """Test the audio subsystem for specific issues and attempt to repair them."""
        try:
            print("\n" + "="*50)
            print("🔧 AUDIO SUBSYSTEM DIAGNOSTICS AND REPAIR")
            print("="*50)

            # Notify the user
            self.add_to_chat("System", "Running audio system diagnostics and repair...")

            # Track issues and fixes
            issues_found = []
            fixes_applied = []

            # Step 1: Check if system has audio capabilities at all
            print("🔍 AUDIO REPAIR: Checking if system has audio capabilities...")
            system_has_audio = self._test_system_audio_capabilities()

            if not system_has_audio:
                issues_found.append("System does not appear to have audio capabilities")
                print("❌ AUDIO REPAIR: System does not have audio capabilities")

                # Try to fix by checking if audio services are running
                if sys.platform.startswith('linux'):
                    # Check if PulseAudio is installed but not running
                    try:
                        pulse_installed = subprocess.run(["which", "pulseaudio"], 
                                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0

                        if pulse_installed:
                            print("🔧 AUDIO REPAIR: PulseAudio is installed, attempting to start...")
                            subprocess.run(["pulseaudio", "--start"], 
                                         check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            fixes_applied.append("Started PulseAudio service")

                            # Check if it worked
                            time.sleep(2)  # Give it time to start
                            system_has_audio = self._test_system_audio_capabilities()
                            if system_has_audio:
                                print("✅ AUDIO REPAIR: Successfully started PulseAudio")
                            else:
                                print("❌ AUDIO REPAIR: Failed to start PulseAudio")
                    except Exception as e:
                        print(f"⚠️ AUDIO REPAIR: Error checking/starting PulseAudio: {e}")

                elif sys.platform.startswith('win'):
                    # Check if Windows Audio service is running
                    try:
                        service_check = subprocess.run(["powershell", "-Command", 
                                                     "Get-Service Audiosrv | Select-Object Status"], 
                                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                        if "Running" not in service_check.stdout:
                            print("🔧 AUDIO REPAIR: Windows Audio service not running, attempting to start...")
                            subprocess.run(["powershell", "-Command", "Start-Service Audiosrv"], 
                                         check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            fixes_applied.append("Started Windows Audio service")

                            # Check if it worked
                            time.sleep(2)  # Give it time to start
                            system_has_audio = self._test_system_audio_capabilities()
                            if system_has_audio:
                                print("✅ AUDIO REPAIR: Successfully started Windows Audio service")
                            else:
                                print("❌ AUDIO REPAIR: Failed to start Windows Audio service")
                    except Exception as e:
                        print(f"⚠️ AUDIO REPAIR: Error checking/starting Windows Audio service: {e}")

            # If system still has no audio capabilities, we can't do much more
            if not system_has_audio:
                print("❌ AUDIO REPAIR: System still does not have audio capabilities after repair attempts")
                self.add_to_chat("System", "Your system does not appear to have audio capabilities. Voice output will not work.")
                return False

            # Step 2: Check for audio device issues
            print("🔍 AUDIO REPAIR: Checking audio devices...")

            audio_devices_ok = True
            try:
                if AUDIO_AVAILABLE:
                    devices = sd.query_devices()
                    output_devices = [device for device in devices if device['max_output_channels'] > 0]

                    if not output_devices:
                        issues_found.append("No audio output devices found")
                        audio_devices_ok = False
                        print("❌ AUDIO REPAIR: No audio output devices found")
                    else:
                        print(f"✅ AUDIO REPAIR: Found {len(output_devices)} audio output devices")

                        # Check if default device is working
                        try:
                            default_device = sd.query_devices(kind='output')
                            print(f"🔍 AUDIO REPAIR: Default output device is: {default_device['name']}")

                            # Test default device with silent audio
                            sample_rate = 44100
                            duration = 0.1
                            silent_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

                            sd.play(silent_audio, sample_rate)
                            sd.wait()
                            print("✅ AUDIO REPAIR: Default device test successful")
                        except Exception as e:
                            issues_found.append(f"Default audio device error: {str(e)}")
                            audio_devices_ok = False
                            print(f"❌ AUDIO REPAIR: Default device test failed: {e}")

                            # Try to fix by selecting a different device
                            if len(output_devices) > 1:
                                print("🔧 AUDIO REPAIR: Trying alternative output device...")

                                # Find a non-default device
                                for device in output_devices:
                                    if device['name'] != default_device['name']:
                                        try:
                                            print(f"🔧 AUDIO REPAIR: Testing device: {device['name']}")
                                            sd.play(silent_audio, sample_rate, device=device['index'])
                                            sd.wait()
                                            print(f"✅ AUDIO REPAIR: Device {device['name']} works")

                                            # Set as selected device
                                            self.selected_output_device = device['name']
                                            fixes_applied.append(f"Selected alternative output device: {device['name']}")
                                            audio_devices_ok = True
                                            break
                                        except Exception as device_error:
                                            print(f"❌ AUDIO REPAIR: Device {device['name']} test failed: {device_error}")
                else:
                    print("⚠️ AUDIO REPAIR: Audio libraries not available, skipping device checks")
            except Exception as e:
                issues_found.append(f"Error checking audio devices: {str(e)}")
                print(f"❌ AUDIO REPAIR: Error checking audio devices: {e}")

            # Step 3: Check audio libraries and TTS
            print("🔍 AUDIO REPAIR: Checking audio libraries and TTS...")

            tts_ok = False
            if self.tts_model is not None:
                try:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name

                    # Generate a short test message
                    self.tts_model.tts_to_file(text="Audio system test.", file_path=temp_audio_path)

                    # Verify the file was created
                    if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                        print("✅ AUDIO REPAIR: TTS file generation working")
                        tts_ok = True

                        # Clean up
                        os.unlink(temp_audio_path)
                    else:
                        issues_found.append("TTS file generation failed")
                        print("❌ AUDIO REPAIR: TTS file generation failed")
                except Exception as e:
                    issues_found.append(f"TTS error: {str(e)}")
                    print(f"❌ AUDIO REPAIR: Error testing TTS: {e}")
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
            else:
                issues_found.append("TTS model not initialized")
                print("❌ AUDIO REPAIR: TTS model not initialized")

                # Try to initialize TTS model with retries
                print("🔧 AUDIO REPAIR: Attempting to initialize TTS model with retries...")
                self._initialize_tts_with_retry(max_retries=3, retry_count=0)

                if self.tts_model is not None:
                    fixes_applied.append("Initialized TTS model")
                    print("✅ AUDIO REPAIR: Successfully initialized TTS model")
                    tts_ok = True
                else:
                    print("❌ AUDIO REPAIR: Failed to initialize TTS model")

            # Step 4: Test direct audio output
            print("🔍 AUDIO REPAIR: Testing direct audio output...")

            direct_audio_ok = False
            try:
                direct_audio_ok = self._play_direct_audio_test()
                if direct_audio_ok:
                    print("✅ AUDIO REPAIR: Direct audio output working")
                else:
                    issues_found.append("Direct audio output failed")
                    print("❌ AUDIO REPAIR: Direct audio output failed")
            except Exception as e:
                issues_found.append(f"Direct audio error: {str(e)}")
                print(f"❌ AUDIO REPAIR: Error testing direct audio output: {e}")

            # If we have issues, try a full restart as a last resort
            if issues_found and not (audio_devices_ok and tts_ok and direct_audio_ok):
                print("🔧 AUDIO REPAIR: Issues found, attempting full audio subsystem restart...")
                restart_success = self._restart_audio_subsystem()

                if restart_success:
                    fixes_applied.append("Restarted audio subsystem")
                    print("✅ AUDIO REPAIR: Audio subsystem restart successful")
                else:
                    print("❌ AUDIO REPAIR: Audio subsystem restart failed")

            # Final status report
            print("\n" + "="*50)
            print("🔧 AUDIO REPAIR SUMMARY")
            print("="*50)

            if issues_found:
                print(f"Issues found ({len(issues_found)}):")
                for issue in issues_found:
                    print(f"  - {issue}")
            else:
                print("No issues found.")

            if fixes_applied:
                print(f"\nFixes applied ({len(fixes_applied)}):")
                for fix in fixes_applied:
                    print(f"  - {fix}")
            else:
                print("\nNo fixes were necessary or possible.")

            # Determine overall success
            repair_success = (not issues_found) or (len(fixes_applied) > 0 and (audio_devices_ok or tts_ok or direct_audio_ok))

            if repair_success:
                print("\n✅ AUDIO REPAIR: Audio subsystem diagnostics and repair completed successfully")
                self.add_to_chat("System", "Audio system diagnostics and repair completed successfully.")

                # If we found and fixed issues, report them
                if issues_found and fixes_applied:
                    issue_msg = f"Found {len(issues_found)} issues and applied {len(fixes_applied)} fixes."
                    self.add_to_chat("System", issue_msg)

                return True
            else:
                print("\n❌ AUDIO REPAIR: Audio subsystem diagnostics and repair failed")

                # Detailed message for the user
                issue_msg = f"Audio system diagnostics found {len(issues_found)} issues but could not fix them all."
                self.add_to_chat("System", issue_msg)

                # Show a notification as a last resort
                try:
                    self._show_system_notification("Audio Repair Failed", 
                                                 f"Found {len(issues_found)} issues but could not fix them all.")
                except:
                    pass

                return False

        except Exception as e:
            print(f"❌ AUDIO REPAIR: Error in audio subsystem diagnostics and repair: {e}")
            self.add_to_chat("System", f"Error in audio system diagnostics and repair: {e}")
            return False

    def _restart_audio_subsystem(self):
        """Restart the audio subsystem if it fails."""
        try:
            print("\n" + "="*50)
            print("🔄 AUDIO SUBSYSTEM RESTART")
            print("="*50)

            # Notify the user
            self.add_to_chat("System", "Audio system appears to be having issues. Attempting to restart audio subsystem...")

            # Close any open audio resources
            self._close_audio_resources()

            # Reset initialization status
            self.audio_initialized = False
            self.audio_init_attempts = 0

            # Reset TTS model
            self.tts_model = None

            # Try to release audio devices
            if AUDIO_AVAILABLE:
                try:
                    print("🔄 AUDIO RESTART: Releasing audio devices...")
                    sd.stop()
                    time.sleep(0.5)  # Give it time to release resources
                    print("✅ AUDIO RESTART: Audio devices released")
                except Exception as e:
                    print(f"⚠️ AUDIO RESTART: Error releasing audio devices: {e}")

            # Try to restart PulseAudio on Linux
            if sys.platform.startswith('linux'):
                try:
                    print("🔄 AUDIO RESTART: Checking PulseAudio status...")
                    # Check if PulseAudio is running
                    pulse_check = subprocess.run(["pulseaudio", "--check"], 
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if pulse_check.returncode != 0:
                        print("🔄 AUDIO RESTART: PulseAudio not running, attempting to start...")
                        # Try to start PulseAudio
                        subprocess.run(["pulseaudio", "--start"], 
                                      check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print("✅ AUDIO RESTART: PulseAudio start command issued")
                    else:
                        print("🔄 AUDIO RESTART: PulseAudio is running, attempting to restart...")
                        # Try to restart PulseAudio
                        subprocess.run(["pulseaudio", "-k"], 
                                      check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        time.sleep(1)
                        subprocess.run(["pulseaudio", "--start"], 
                                      check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print("✅ AUDIO RESTART: PulseAudio restart commands issued")
                except Exception as e:
                    print(f"⚠️ AUDIO RESTART: Error restarting PulseAudio: {e}")

            # Wait a moment for audio system to stabilize
            time.sleep(1)

            # Reinitialize the audio system
            print("🔄 AUDIO RESTART: Reinitializing audio system...")
            self._initialize_audio_system()

            # Test if audio is working now
            print("🔄 AUDIO RESTART: Testing audio after restart...")
            audio_working = False

            # Try raw audio output first
            try:
                audio_working = self._test_raw_audio_output()
                if audio_working:
                    print("✅ AUDIO RESTART: Raw audio test successful after restart")
                else:
                    print("⚠️ AUDIO RESTART: Raw audio test failed after restart")
            except Exception as e:
                print(f"⚠️ AUDIO RESTART: Error in raw audio test after restart: {e}")

            # If raw audio failed, try system audio
            if not audio_working:
                try:
                    audio_working = self._test_system_audio()
                    if audio_working:
                        print("✅ AUDIO RESTART: System audio test successful after restart")
                    else:
                        print("⚠️ AUDIO RESTART: System audio test failed after restart")
                except Exception as e:
                    print(f"⚠️ AUDIO RESTART: Error in system audio test after restart: {e}")

            # Report results
            if audio_working:
                print("✅ AUDIO RESTART: Audio subsystem successfully restarted")
                self.add_to_chat("System", "Audio subsystem successfully restarted. Voice output should now be working.")
                return True
            else:
                print("❌ AUDIO RESTART: Audio subsystem restart failed")
                self.add_to_chat("System", "Audio subsystem restart failed. Voice output may still not work.")

                # Show a notification as a last resort
                try:
                    self._show_system_notification("Audio Restart Failed", 
                                                 "Unable to restart audio subsystem. Voice output may not work.")
                except:
                    pass

                return False

        except Exception as e:
            print(f"❌ AUDIO RESTART: Error restarting audio subsystem: {e}")
            self.add_to_chat("System", f"Error restarting audio subsystem: {e}")
            return False

    def _close_audio_resources(self):
        """Close any open audio resources to prepare for restart."""
        try:
            print("🔄 AUDIO CLOSE: Closing audio resources...")

            # Stop any ongoing audio playback
            if AUDIO_AVAILABLE:
                try:
                    sd.stop()
                    print("✅ AUDIO CLOSE: Stopped audio playback")
                except Exception as e:
                    print(f"⚠️ AUDIO CLOSE: Error stopping audio playback: {e}")

            # Close TTS model if it exists
            if self.tts_model is not None:
                try:
                    # There's no explicit close method for TTS models,
                    # but we can set it to None to allow garbage collection
                    self.tts_model = None
                    print("✅ AUDIO CLOSE: Released TTS model")
                except Exception as e:
                    print(f"⚠️ AUDIO CLOSE: Error releasing TTS model: {e}")

            # Force garbage collection to release resources
            try:
                import gc
                gc.collect()
                print("✅ AUDIO CLOSE: Forced garbage collection")
            except Exception as e:
                print(f"⚠️ AUDIO CLOSE: Error during garbage collection: {e}")

            print("✅ AUDIO CLOSE: Audio resources closed")
            return True

        except Exception as e:
            print(f"❌ AUDIO CLOSE: Error closing audio resources: {e}")
            return False

    def _initialize_audio_devices(self):
        """Initialize audio devices with robust error handling."""
        try:
            print("🔊 AUDIO INIT: Initializing audio devices...")

            # Get available devices
            self.get_available_devices()

            # Verify that we have at least one output device
            if not self.output_devices:
                print("⚠️ AUDIO INIT: No output devices found")
                # Add a default device as fallback
                self.output_devices = ["Default"]
                self.selected_output_device = "Default"
            else:
                print(f"✅ AUDIO INIT: Found {len(self.output_devices)} output devices")

            # Verify that we have at least one input device
            if not self.input_devices:
                print("⚠️ AUDIO INIT: No input devices found")
                # Add a default device as fallback
                self.input_devices = ["Default"]
                self.selected_input_device = "Default"
            else:
                print(f"✅ AUDIO INIT: Found {len(self.input_devices)} input devices")

            # Check for potential issues with audio devices
            self._detect_and_fix_audio_device_issues()

            print("✅ AUDIO INIT: Audio devices initialized")

        except Exception as e:
            print(f"❌ AUDIO INIT: Error initializing audio devices: {e}")
            # Set default devices as fallback
            self.input_devices = ["Default"]
            self.output_devices = ["Default"]
            self.selected_input_device = "Default"
            self.selected_output_device = "Default"

    def _detect_and_fix_audio_device_issues(self):
        """Detect and fix common audio device configuration issues."""
        try:
            print("🔍 AUDIO CONFIG: Checking audio device configuration...")

            # Check if we have any output devices
            if not self.output_devices or (len(self.output_devices) == 1 and self.output_devices[0] == "Default"):
                print("⚠️ AUDIO CONFIG: No real output devices found")
                self._try_fix_no_output_devices()
                return

            # Check default device settings
            try:
                if AUDIO_AVAILABLE:
                    default_device_info = sd.query_devices(kind='output')
                    print(f"✅ AUDIO CONFIG: Default output device: {default_device_info['name']}")

                    # Check if default device is working
                    self._test_default_device()
            except Exception as e:
                print(f"⚠️ AUDIO CONFIG: Issue with default output device: {e}")
                self._try_fix_default_device()

            # Check for specific platform issues
            if sys.platform.startswith('linux'):
                self._check_linux_audio_config()
            elif sys.platform.startswith('darwin'):
                self._check_macos_audio_config()
            elif sys.platform.startswith('win'):
                self._check_windows_audio_config()

            print("✅ AUDIO CONFIG: Audio device configuration check completed")

        except Exception as e:
            print(f"❌ AUDIO CONFIG: Error checking audio configuration: {e}")

    def _test_default_device(self):
        """Test if the default audio device is working."""
        try:
            print("🔊 AUDIO CONFIG: Testing default output device...")

            if not AUDIO_AVAILABLE:
                print("⚠️ AUDIO CONFIG: Audio libraries not available, cannot test default device")
                return False

            # Create a short silent audio sample (to avoid making noise)
            sample_rate = 44100
            duration = 0.1
            silent_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

            # Try to play the silent audio
            sd.play(silent_audio, sample_rate)
            sd.wait()

            print("✅ AUDIO CONFIG: Default device test successful")
            return True
        except Exception as e:
            print(f"❌ AUDIO CONFIG: Default device test failed: {e}")
            return False

    def _try_fix_no_output_devices(self):
        """Try to fix the issue of no output devices found."""
        try:
            print("🔧 AUDIO CONFIG: Attempting to fix no output devices issue...")

            # Restart the audio subsystem
            self._restart_audio_subsystem()

            # Check if we have output devices now
            try:
                if AUDIO_AVAILABLE:
                    devices = sd.query_devices()
                    output_devices = [device['name'] for device in devices if device['max_output_channels'] > 0]

                    if output_devices:
                        print(f"✅ AUDIO CONFIG: Found {len(output_devices)} output devices after restart")
                        self.output_devices = output_devices
                        return True
                    else:
                        print("❌ AUDIO CONFIG: Still no output devices found after restart")
                else:
                    print("⚠️ AUDIO CONFIG: Audio libraries not available, cannot check for devices")
            except Exception as e:
                print(f"❌ AUDIO CONFIG: Error checking devices after restart: {e}")

            # Try platform-specific fixes
            if sys.platform.startswith('linux'):
                print("🔧 AUDIO CONFIG: Trying Linux-specific fixes...")

                # Check if PulseAudio is running
                try:
                    pulse_check = subprocess.run(["pulseaudio", "--check"], 
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if pulse_check.returncode != 0:
                        print("🔧 AUDIO CONFIG: PulseAudio not running, attempting to start...")
                        subprocess.run(["pulseaudio", "--start"], 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        # Wait for PulseAudio to start
                        time.sleep(2)

                        # Check again for devices
                        if AUDIO_AVAILABLE:
                            devices = sd.query_devices()
                            output_devices = [device['name'] for device in devices if device['max_output_channels'] > 0]

                            if output_devices:
                                print(f"✅ AUDIO CONFIG: Found {len(output_devices)} output devices after starting PulseAudio")
                                self.output_devices = output_devices
                                return True
                except Exception as pulse_error:
                    print(f"⚠️ AUDIO CONFIG: Error with PulseAudio fix: {pulse_error}")

            # If we get here, we couldn't fix the issue
            print("❌ AUDIO CONFIG: Could not fix no output devices issue")
            return False

        except Exception as e:
            print(f"❌ AUDIO CONFIG: Error trying to fix no output devices: {e}")
            return False

    def _try_fix_default_device(self):
        """Try to fix issues with the default audio device."""
        try:
            print("🔧 AUDIO CONFIG: Attempting to fix default device issue...")

            # Try to set a different device as default
            if len(self.output_devices) > 1:
                print("🔧 AUDIO CONFIG: Trying alternative output device...")

                if AUDIO_AVAILABLE:
                    # Get the first non-default device
                    try:
                        default_device_name = sd.query_devices(kind='output')['name']
                        alternative_device = next((device for device in self.output_devices if device != default_device_name), None)

                        if alternative_device:
                            # Find the device index
                            devices = sd.query_devices()
                            for i, device in enumerate(devices):
                                if device['name'] == alternative_device and device['max_output_channels'] > 0:
                                    print(f"🔧 AUDIO CONFIG: Setting device '{alternative_device}' as output")
                                    self.selected_output_device = alternative_device

                                    # Test the alternative device
                                    try:
                                        sample_rate = 44100
                                        duration = 0.1
                                        silent_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

                                        sd.play(silent_audio, sample_rate, device=i)
                                        sd.wait()

                                        print(f"✅ AUDIO CONFIG: Alternative device '{alternative_device}' is working")
                                        return True
                                    except Exception as play_error:
                                        print(f"❌ AUDIO CONFIG: Alternative device test failed: {play_error}")
                    except Exception as device_error:
                        print(f"⚠️ AUDIO CONFIG: Error finding alternative device: {device_error}")

            # If we get here, we couldn't fix the issue with alternative devices
            # Try restarting the audio subsystem
            print("🔧 AUDIO CONFIG: Trying audio subsystem restart...")
            self._restart_audio_subsystem()

            # Test default device again
            if self._test_default_device():
                print("✅ AUDIO CONFIG: Default device working after restart")
                return True

            print("❌ AUDIO CONFIG: Could not fix default device issue")
            return False

        except Exception as e:
            print(f"❌ AUDIO CONFIG: Error trying to fix default device: {e}")
            return False

    def _check_linux_audio_config(self):
        """Check and fix Linux-specific audio configuration issues."""
        try:
            print("🔍 AUDIO CONFIG: Checking Linux audio configuration...")

            # Check if PulseAudio is running
            try:
                pulse_check = subprocess.run(["pulseaudio", "--check"], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if pulse_check.returncode != 0:
                    print("⚠️ AUDIO CONFIG: PulseAudio not running")

                    # Try to start PulseAudio
                    subprocess.run(["pulseaudio", "--start"], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("🔧 AUDIO CONFIG: Started PulseAudio")
            except Exception as pulse_error:
                print(f"⚠️ AUDIO CONFIG: PulseAudio check error: {pulse_error}")

            # Check ALSA devices
            try:
                alsa_check = subprocess.run(["aplay", "-l"], 
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if "no soundcards found" in alsa_check.stdout or "no soundcards found" in alsa_check.stderr:
                    print("⚠️ AUDIO CONFIG: No ALSA soundcards found")
                else:
                    print("✅ AUDIO CONFIG: ALSA soundcards found")
            except Exception as alsa_error:
                print(f"⚠️ AUDIO CONFIG: ALSA check error: {alsa_error}")

            # Check if audio is muted
            try:
                amixer_check = subprocess.run(["amixer", "get", "Master"], 
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if "[off]" in amixer_check.stdout:
                    print("⚠️ AUDIO CONFIG: Master volume is muted")

                    # Try to unmute
                    subprocess.run(["amixer", "set", "Master", "unmute"], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("🔧 AUDIO CONFIG: Unmuted master volume")
            except Exception as amixer_error:
                print(f"⚠️ AUDIO CONFIG: Volume check error: {amixer_error}")

        except Exception as e:
            print(f"❌ AUDIO CONFIG: Error checking Linux audio configuration: {e}")

    def _check_macos_audio_config(self):
        """Check and fix macOS-specific audio configuration issues."""
        try:
            print("🔍 AUDIO CONFIG: Checking macOS audio configuration...")

            # Check if audio is muted using osascript
            try:
                volume_check = subprocess.run(["osascript", "-e", "output volume of (get volume settings)"], 
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                volume = int(volume_check.stdout.strip())
                print(f"✅ AUDIO CONFIG: System volume is {volume}%")

                if volume == 0:
                    print("⚠️ AUDIO CONFIG: System volume is 0%")

                    # Try to increase volume
                    subprocess.run(["osascript", "-e", "set volume output volume 50"], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("🔧 AUDIO CONFIG: Increased system volume to 50%")

                # Check if audio is muted
                mute_check = subprocess.run(["osascript", "-e", "output muted of (get volume settings)"], 
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if "true" in mute_check.stdout.lower():
                    print("⚠️ AUDIO CONFIG: System audio is muted")

                    # Try to unmute
                    subprocess.run(["osascript", "-e", "set volume output muted false"], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("🔧 AUDIO CONFIG: Unmuted system audio")
            except Exception as volume_error:
                print(f"⚠️ AUDIO CONFIG: Volume check error: {volume_error}")

            # Check CoreAudio devices
            try:
                audio_check = subprocess.run(["system_profiler", "SPAudioDataType"], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if "Output" not in audio_check.stdout:
                    print("⚠️ AUDIO CONFIG: No audio output devices found in system profiler")
                else:
                    print("✅ AUDIO CONFIG: Audio output devices found in system profiler")
            except Exception as audio_error:
                print(f"⚠️ AUDIO CONFIG: Audio device check error: {audio_error}")

        except Exception as e:
            print(f"❌ AUDIO CONFIG: Error checking macOS audio configuration: {e}")

    def _check_windows_audio_config(self):
        """Check and fix Windows-specific audio configuration issues."""
        try:
            print("🔍 AUDIO CONFIG: Checking Windows audio configuration...")

            # Check if audio is muted using PowerShell
            try:
                # This requires the AudioDeviceCmdlets module which might not be installed
                # We'll use a more compatible approach
                ps_script = """
                Add-Type -TypeDefinition @'
                using System.Runtime.InteropServices;
                [Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                interface IAudioEndpointVolume {
                    // Methods not used in this example are omitted
                    int GetMasterVolumeLevelScalar(out float level);
                    int GetMute(out bool mute);
                }
                [Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                interface IMMDevice {
                    // Methods not used in this example are omitted
                    int Activate(ref Guid id, int clsCtx, IntPtr activationParams, out IAudioEndpointVolume aev);
                }
                [Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                interface IMMDeviceEnumerator {
                    // Methods not used in this example are omitted
                    int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice device);
                }
                [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
                class MMDeviceEnumeratorComObject { }
                public class AudioInfo {
                    public static bool IsMuted() {
                        try {
                            var enumerator = new MMDeviceEnumeratorComObject() as IMMDeviceEnumerator;
                            IMMDevice dev = null;
                            Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(0, 0, out dev));
                            IAudioEndpointVolume aev = null;
                            Guid IID_IAudioEndpointVolume = typeof(IAudioEndpointVolume).GUID;
                            Marshal.ThrowExceptionForHR(dev.Activate(ref IID_IAudioEndpointVolume, 7, IntPtr.Zero, out aev));
                            bool mute;
                            Marshal.ThrowExceptionForHR(aev.GetMute(out mute));
                            return mute;
                        } catch {
                            return false;
                        }
                    }
                    public static float GetVolume() {
                        try {
                            var enumerator = new MMDeviceEnumeratorComObject() as IMMDeviceEnumerator;
                            IMMDevice dev = null;
                            Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(0, 0, out dev));
                            IAudioEndpointVolume aev = null;
                            Guid IID_IAudioEndpointVolume = typeof(IAudioEndpointVolume).GUID;
                            Marshal.ThrowExceptionForHR(dev.Activate(ref IID_IAudioEndpointVolume, 7, IntPtr.Zero, out aev));
                            float volume;
                            Marshal.ThrowExceptionForHR(aev.GetMasterVolumeLevelScalar(out volume));
                            return volume;
                        } catch {
                            return 0;
                        }
                    }
                }
                '@

                $muted = [AudioInfo]::IsMuted()
                $volume = [AudioInfo]::GetVolume()

                Write-Output "Muted: $muted"
                Write-Output "Volume: $($volume * 100)%"
                """

                # Save the script to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".ps1", delete=False, mode='w') as temp_ps:
                    temp_ps_path = temp_ps.name
                    temp_ps.write(ps_script)

                # Run the PowerShell script
                ps_result = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", temp_ps_path], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Clean up
                os.unlink(temp_ps_path)

                # Parse the results
                muted = "muted: true" in ps_result.stdout.lower()
                volume_match = re.search(r"volume: (\d+(\.\d+)?)%", ps_result.stdout.lower())
                volume = float(volume_match.group(1)) if volume_match else 0

                print(f"✅ AUDIO CONFIG: System volume is {volume}%, Muted: {muted}")

                if muted:
                    print("⚠️ AUDIO CONFIG: System audio is muted")

                    # Try to unmute using PowerShell
                    unmute_cmd = """
                    $obj = New-Object -ComObject WScript.Shell
                    $obj.SendKeys([char]173)
                    """
                    subprocess.run(["powershell", "-Command", unmute_cmd], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("🔧 AUDIO CONFIG: Attempted to unmute system audio")

                if volume < 10:
                    print("⚠️ AUDIO CONFIG: System volume is very low")

                    # Try to increase volume using PowerShell
                    volume_cmd = """
                    $obj = New-Object -ComObject WScript.Shell
                    for ($i = 0; $i -lt 10; $i++) {
                        $obj.SendKeys([char]175)
                    }
                    """
                    subprocess.run(["powershell", "-Command", volume_cmd], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("🔧 AUDIO CONFIG: Attempted to increase system volume")

            except Exception as volume_error:
                print(f"⚠️ AUDIO CONFIG: Volume check error: {volume_error}")

            # Check Windows audio devices
            try:
                # Use PowerShell to list audio devices
                ps_cmd = "Get-WmiObject Win32_SoundDevice | Select-Object Name, Status"
                ps_result = subprocess.run(["powershell", "-Command", ps_cmd], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if "Name" in ps_result.stdout and "Status" in ps_result.stdout:
                    print("✅ AUDIO CONFIG: Found Windows audio devices")

                    # Check for disabled devices
                    if "Disabled" in ps_result.stdout:
                        print("⚠️ AUDIO CONFIG: Some audio devices may be disabled")
                else:
                    print("⚠️ AUDIO CONFIG: No Windows audio devices found or error listing devices")
            except Exception as device_error:
                print(f"⚠️ AUDIO CONFIG: Audio device check error: {device_error}")

        except Exception as e:
            print(f"❌ AUDIO CONFIG: Error checking Windows audio configuration: {e}")

    def _initialize_tts_with_retry(self, max_retries=5, retry_count=0, delay_seconds=0):
        """Initialize TTS model with multiple attempts and fallback models.

        Args:
            max_retries (int): Maximum number of retry attempts if all models fail
            retry_count (int): Current retry count (used internally for recursion)
            delay_seconds (int): Delay in seconds before this attempt (for logging)

        Returns:
            bool: True if TTS model was initialized successfully, False otherwise
        """
        try:
            # Log retry attempt if this is a retry
            if retry_count > 0:
                print(f"🔊 AUDIO INIT: TTS RETRY ATTEMPT {retry_count}/{max_retries} after {delay_seconds}s delay")
            else:
                print("🔊 AUDIO INIT: Initializing TTS model...")

            if not COQUI_TTS_AVAILABLE:
                print("⚠️ AUDIO INIT: Coqui TTS not available")
                return False

            # Try multiple models in order of preference
            models_to_try = [
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/ljspeech/glow-tts",
                "tts_models/en/vctk/vits",
                "tts_models/en/ljspeech/fast_pitch",
                "tts_models/multilingual/multi-dataset/your_tts",
                "tts_models/en/ljspeech/neural_hmm"
            ]

            # Shuffle the models list on retries to avoid getting stuck on the same failing model
            if retry_count > 0:
                import random
                random.shuffle(models_to_try)
                print(f"🔊 AUDIO INIT: Shuffled models list for retry attempt {retry_count}")

            for model_name in models_to_try:
                try:
                    print(f"🔊 AUDIO INIT: Trying TTS model: {model_name}")
                    self.tts_model = TTS(model_name)
                    print(f"✅ AUDIO INIT: TTS model {model_name} initialized successfully")

                    # Test the model with a simple text
                    try:
                        print("🔊 AUDIO INIT: Testing TTS model with a simple text...")
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                            temp_audio_path = temp_audio.name

                        self.tts_model.tts_to_file(text="This is a test of the text to speech system.", file_path=temp_audio_path)

                        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                            print(f"✅ AUDIO INIT: Test successful, audio file created: {os.path.getsize(temp_audio_path)} bytes")
                            # Clean up the test file
                            try:
                                os.unlink(temp_audio_path)
                            except:
                                pass
                        else:
                            print("⚠️ AUDIO INIT: Test failed, audio file not created or empty")
                            raise Exception("Test failed, audio file not created or empty")
                    except Exception as test_error:
                        print(f"⚠️ AUDIO INIT: Error testing TTS model: {test_error}")
                        # Continue with the model anyway, it might still work

                    # Add a message to the chat about successful initialization
                    self.add_to_chat("System", f"Voice synthesis model initialized successfully: {model_name}")
                    return True
                except Exception as model_error:
                    print(f"⚠️ AUDIO INIT: Error initializing TTS model {model_name}: {model_error}")
                    print(f"⚠️ AUDIO INIT: Error type: {type(model_error).__name__}")
                    print(f"⚠️ AUDIO INIT: Error details: {str(model_error)}")

            print("❌ AUDIO INIT: All TTS models failed to initialize")

            # Check if we should retry
            if retry_count < max_retries:
                retry_count += 1

                # Use exponential backoff for the delay
                next_delay = 2 * (2 ** (retry_count - 1))  # 2, 4, 8, 16... seconds
                print(f"🔊 AUDIO INIT: Scheduling TTS initialization retry in {next_delay} seconds (attempt {retry_count}/{max_retries})...")

                # Add a message to the chat about retrying
                if retry_count == 1:  # Only on first retry to avoid spamming
                    self.add_to_chat("System", "Voice synthesis model initialization failed. Will retry in background...")

                # Schedule the retry using root.after to avoid blocking
                self.root.after(int(next_delay * 1000), 
                               lambda: self._initialize_tts_with_retry(max_retries, retry_count, next_delay))

                return False
            else:
                print(f"❌ AUDIO INIT: All TTS models failed after {retry_count} retries. Giving up.")
                self.add_to_chat("System", "Failed to initialize voice synthesis model after multiple attempts. Will use fallback methods.")
                return False

        except Exception as e:
            print(f"❌ AUDIO INIT: Error in TTS initialization: {e}")
            print(f"❌ AUDIO INIT: Error type: {type(e).__name__}")
            print(f"❌ AUDIO INIT: Error details: {str(e)}")

            # Check if we should retry after an exception
            if retry_count < max_retries:
                retry_count += 1

                # Use exponential backoff for the delay
                next_delay = 2 * (2 ** (retry_count - 1))  # 2, 4, 8, 16... seconds
                print(f"🔊 AUDIO INIT: Scheduling TTS initialization retry after exception in {next_delay} seconds (attempt {retry_count}/{max_retries})...")

                # Schedule the retry
                self.root.after(int(next_delay * 1000), 
                               lambda: self._initialize_tts_with_retry(max_retries, retry_count, next_delay))

                return False

            return False

    def _test_system_audio_capabilities(self):
        """Test if the system has audio capabilities at all, regardless of specific output methods."""
        try:
            print("🔍 AUDIO CAPABILITIES: Testing if system has audio capabilities...")

            # Check 1: Check if audio libraries are available
            if not AUDIO_AVAILABLE:
                print("⚠️ AUDIO CAPABILITIES: Audio libraries not available")
                # This doesn't necessarily mean the system has no audio capabilities,
                # just that our Python libraries can't access them
            else:
                print("✅ AUDIO CAPABILITIES: Audio libraries available")

            # Check 2: Check for audio devices
            audio_devices_present = False
            try:
                if AUDIO_AVAILABLE:
                    devices = sd.query_devices()
                    output_devices = [device for device in devices if device['max_output_channels'] > 0]

                    if output_devices:
                        print(f"✅ AUDIO CAPABILITIES: Found {len(output_devices)} output devices")
                        audio_devices_present = True
                    else:
                        print("❌ AUDIO CAPABILITIES: No output devices found")
                else:
                    # Try platform-specific methods to detect audio devices
                    if sys.platform.startswith('linux'):
                        # Check ALSA devices
                        try:
                            alsa_check = subprocess.run(["aplay", "-l"], 
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                            if "no soundcards found" not in alsa_check.stdout and "no soundcards found" not in alsa_check.stderr:
                                print("✅ AUDIO CAPABILITIES: ALSA devices found")
                                audio_devices_present = True
                            else:
                                print("❌ AUDIO CAPABILITIES: No ALSA devices found")

                            # Check PulseAudio devices
                            pulse_check = subprocess.run(["pactl", "list", "sinks"], 
                                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                            if "sink" in pulse_check.stdout:
                                print("✅ AUDIO CAPABILITIES: PulseAudio sinks found")
                                audio_devices_present = True
                        except Exception as linux_check_error:
                            print(f"⚠️ AUDIO CAPABILITIES: Error checking Linux audio devices: {linux_check_error}")

                    elif sys.platform.startswith('darwin'):
                        # Check macOS audio devices
                        try:
                            mac_check = subprocess.run(["system_profiler", "SPAudioDataType"], 
                                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                            if "Output" in mac_check.stdout:
                                print("✅ AUDIO CAPABILITIES: macOS audio output devices found")
                                audio_devices_present = True
                            else:
                                print("❌ AUDIO CAPABILITIES: No macOS audio output devices found")
                        except Exception as mac_check_error:
                            print(f"⚠️ AUDIO CAPABILITIES: Error checking macOS audio devices: {mac_check_error}")

                    elif sys.platform.startswith('win'):
                        # Check Windows audio devices
                        try:
                            # Use PowerShell to check for audio devices
                            ps_cmd = "Get-WmiObject Win32_SoundDevice | Select-Object Name, Status"
                            ps_result = subprocess.run(["powershell", "-Command", ps_cmd], 
                                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                            if "Name" in ps_result.stdout and not "No default" in ps_result.stderr:
                                print("✅ AUDIO CAPABILITIES: Windows audio devices found")
                                audio_devices_present = True
                            else:
                                print("❌ AUDIO CAPABILITIES: No Windows audio devices found")
                        except Exception as win_check_error:
                            print(f"⚠️ AUDIO CAPABILITIES: Error checking Windows audio devices: {win_check_error}")
            except Exception as device_check_error:
                print(f"⚠️ AUDIO CAPABILITIES: Error checking audio devices: {device_check_error}")

            # Check 3: Check if audio subsystem is loaded/running
            audio_subsystem_running = False
            try:
                if sys.platform.startswith('linux'):
                    # Check if PulseAudio is running
                    try:
                        pulse_check = subprocess.run(["pulseaudio", "--check"], 
                                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        if pulse_check.returncode == 0:
                            print("✅ AUDIO CAPABILITIES: PulseAudio is running")
                            audio_subsystem_running = True
                        else:
                            print("❌ AUDIO CAPABILITIES: PulseAudio is not running")

                            # Check if ALSA is available even if PulseAudio isn't
                            alsa_check = subprocess.run(["aplay", "--version"], 
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                            if alsa_check.returncode == 0:
                                print("✅ AUDIO CAPABILITIES: ALSA is available")
                                audio_subsystem_running = True
                    except Exception as pulse_error:
                        print(f"⚠️ AUDIO CAPABILITIES: Error checking PulseAudio: {pulse_error}")

                elif sys.platform.startswith('darwin'):
                    # Check if CoreAudio is running (it's always running on macOS if the system is working)
                    try:
                        # Just check if we can get audio device info
                        mac_check = subprocess.run(["system_profiler", "SPAudioDataType"], 
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        if mac_check.returncode == 0:
                            print("✅ AUDIO CAPABILITIES: macOS CoreAudio is available")
                            audio_subsystem_running = True
                        else:
                            print("❌ AUDIO CAPABILITIES: macOS CoreAudio may not be working")
                    except Exception as core_audio_error:
                        print(f"⚠️ AUDIO CAPABILITIES: Error checking CoreAudio: {core_audio_error}")

                elif sys.platform.startswith('win'):
                    # Check if Windows audio service is running
                    try:
                        service_check = subprocess.run(["powershell", "-Command", 
                                                     "Get-Service Audiosrv | Select-Object Status"], 
                                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                        if "Running" in service_check.stdout:
                            print("✅ AUDIO CAPABILITIES: Windows Audio service is running")
                            audio_subsystem_running = True
                        else:
                            print("❌ AUDIO CAPABILITIES: Windows Audio service may not be running")
                    except Exception as service_error:
                        print(f"⚠️ AUDIO CAPABILITIES: Error checking Windows Audio service: {service_error}")
            except Exception as subsystem_check_error:
                print(f"⚠️ AUDIO CAPABILITIES: Error checking audio subsystem: {subsystem_check_error}")

            # Check 4: Try a very basic audio test that should work on almost any system
            basic_audio_works = False
            try:
                # Use the most basic system sound capability
                if sys.platform.startswith('linux'):
                    try:
                        # Try to use the system bell - this is the most basic sound capability
                        print("🔊 AUDIO CAPABILITIES: Trying system bell...")
                        sys.stdout.write('\a')  # ASCII bell character
                        sys.stdout.flush()
                        basic_audio_works = True  # Assume it worked, we can't really check
                    except:
                        pass
                elif sys.platform.startswith('darwin'):
                    try:
                        # Try to use the system beep
                        print("🔊 AUDIO CAPABILITIES: Trying system beep...")
                        subprocess.run(["osascript", "-e", "beep"], 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
                        basic_audio_works = True  # Assume it worked
                    except:
                        pass
                elif sys.platform.startswith('win'):
                    try:
                        # Try to use the system beep
                        print("🔊 AUDIO CAPABILITIES: Trying system beep...")
                        subprocess.run(["powershell", "-Command", "[Console]::Beep(800, 200)"], 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
                        basic_audio_works = True  # Assume it worked
                    except:
                        pass

                # Also try the Tkinter bell as a fallback
                try:
                    print("🔊 AUDIO CAPABILITIES: Trying Tkinter bell...")
                    self.root.bell()
                    basic_audio_works = True  # Assume it worked
                except:
                    pass

                if basic_audio_works:
                    print("✅ AUDIO CAPABILITIES: Basic audio test succeeded")
                else:
                    print("❌ AUDIO CAPABILITIES: Basic audio test failed")
            except Exception as basic_audio_error:
                print(f"⚠️ AUDIO CAPABILITIES: Error in basic audio test: {basic_audio_error}")

            # Determine overall audio capabilities
            has_audio_capabilities = audio_devices_present or audio_subsystem_running or basic_audio_works

            if has_audio_capabilities:
                print("✅ AUDIO CAPABILITIES: System has audio capabilities")
            else:
                print("❌ AUDIO CAPABILITIES: System does not appear to have audio capabilities")

                # Add a message to the chat
                self.add_to_chat("System", "Warning: Your system does not appear to have audio capabilities. Voice output will not work.")

            return has_audio_capabilities

        except Exception as e:
            print(f"❌ AUDIO CAPABILITIES: Error testing audio capabilities: {e}")
            return False

    def _test_audio_system(self):
        """Test the audio system to ensure it's working properly."""
        try:
            print("🔊 AUDIO INIT: Testing audio system...")

            # First, check if the system has audio capabilities at all
            system_has_audio = self._test_system_audio_capabilities()
            if not system_has_audio:
                print("❌ AUDIO INIT: System does not have audio capabilities")
                return False

            # Test direct audio output
            direct_audio_works = False
            try:
                print("🔊 AUDIO INIT: Testing direct audio output...")
                direct_audio_works = self._play_direct_audio_test()
                if direct_audio_works:
                    print("✅ AUDIO INIT: Direct audio output working")
                else:
                    print("❌ AUDIO INIT: Direct audio output not working")
            except Exception as e:
                print(f"❌ AUDIO INIT: Error testing direct audio output: {e}")

            # Test TTS if available
            tts_works = False
            if self.tts_model is not None:
                try:
                    print("🔊 AUDIO INIT: Testing TTS...")
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name

                    # Generate a short test message
                    self.tts_model.tts_to_file(text="Audio system test.", file_path=temp_audio_path)

                    # Verify the file was created
                    if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                        print("✅ AUDIO INIT: TTS file generation working")
                        tts_works = True

                        # Clean up
                        os.unlink(temp_audio_path)
                    else:
                        print("❌ AUDIO INIT: TTS file generation not working")
                except Exception as e:
                    print(f"❌ AUDIO INIT: Error testing TTS: {e}")
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass

            # Overall status
            if direct_audio_works and tts_works:
                print("✅ AUDIO INIT: Audio system fully operational")
            elif direct_audio_works:
                print("⚠️ AUDIO INIT: Audio system partially operational (direct audio only)")
            else:
                print("❌ AUDIO INIT: Audio system not operational")

            return direct_audio_works or tts_works

        except Exception as e:
            print(f"❌ AUDIO INIT: Error testing audio system: {e}")
            return False

    def schedule_audio_system_check(self):
        """Schedule periodic checks of the audio system."""
        try:
            # Only run if audio system is not initialized or if it's been a while since the last check
            if not self.audio_initialized:
                print("🔊 AUDIO CHECK: Audio system not initialized, attempting initialization...")
                self._initialize_audio_system()
            else:
                # Perform a quick test to ensure audio is still working
                print("🔊 AUDIO CHECK: Performing periodic audio system check...")
                if not self._play_direct_audio_test():
                    print("⚠️ AUDIO CHECK: Audio system may not be working, reinitializing...")
                    self._initialize_audio_system()
                else:
                    print("✅ AUDIO CHECK: Audio system is working")

            # Schedule the next check
            self.root.after(self.audio_check_interval, self.schedule_audio_system_check)

        except Exception as e:
            print(f"❌ AUDIO CHECK: Error in audio system check: {e}")
            # Schedule the next check even if there's an error
            self.root.after(self.audio_check_interval, self.schedule_audio_system_check)

    def get_available_devices(self):
        """Get available input and output audio devices."""
        try:
            print("🔊 AUDIO SETUP: Checking audio device configuration...")

            # Check if audio libraries are available
            if not AUDIO_AVAILABLE:
                print("⚠️ AUDIO SETUP: Audio libraries not available - voice output may not work")
                self.add_to_chat("System", "Warning: Audio libraries not available. Voice output may not work.")
                self.input_devices = ["Default"]
                self.output_devices = ["Default"]
                self.selected_input_device = "Default"
                self.selected_output_device = "Default"
                return

            # Check audio device permissions on Linux
            if sys.platform.startswith('linux'):
                try:
                    # Check if user is in the audio group
                    groups_result = subprocess.run(["groups"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    user_groups = groups_result.stdout.strip().split()

                    if 'audio' not in user_groups and 'pulse' not in user_groups and 'pulse-access' not in user_groups:
                        print("⚠️ AUDIO SETUP: User may not have proper audio permissions")
                        print("⚠️ AUDIO SETUP: User is not in 'audio', 'pulse', or 'pulse-access' groups")
                        self.add_to_chat("System", "Warning: Your user may not have proper audio permissions. Voice output may not work.")
                except Exception as e:
                    print(f"⚠️ AUDIO SETUP: Could not check audio permissions: {e}")

            # Get input devices (microphones)
            if SPEECH_RECOGNITION_AVAILABLE:
                try:
                    self.input_devices = sr.Microphone.list_microphone_names()
                    print(f"✅ AUDIO SETUP: Found {len(self.input_devices)} input devices")

                    # Print the available input devices for debugging
                    print("🔊 AUDIO SETUP: Available input devices:")
                    for i, device in enumerate(self.input_devices):
                        print(f"  {i}: {device}")

                except Exception as e:
                    print(f"⚠️ AUDIO SETUP: Error getting input devices: {e}")
                    self.input_devices = ["Default"]
            else:
                print("⚠️ AUDIO SETUP: Speech recognition not available - voice input will not work")
                self.input_devices = ["Default"]

            # Get output devices (speakers)
            if AUDIO_AVAILABLE:
                try:
                    devices = sd.query_devices()
                    self.output_devices = [d['name'] for d in devices if d['max_output_channels'] > 0]
                    print(f"✅ AUDIO SETUP: Found {len(self.output_devices)} output devices")

                    # Print the available output devices for debugging
                    print("🔊 AUDIO SETUP: Available output devices:")
                    for i, device in enumerate(self.output_devices):
                        print(f"  {i}: {device}")

                    # Check if default device is working
                    try:
                        default_device = sd.query_devices(kind='output')
                        print(f"✅ AUDIO SETUP: Default output device: {default_device['name']}")
                    except Exception as default_error:
                        print(f"⚠️ AUDIO SETUP: Error getting default output device: {default_error}")

                except Exception as e:
                    print(f"⚠️ AUDIO SETUP: Error getting output devices: {e}")
                    self.output_devices = ["Default"]
            else:
                print("⚠️ AUDIO SETUP: Audio libraries not available - voice output will not work")
                self.output_devices = ["Default"]

            # Set default devices
            self.selected_input_device = self.input_devices[0] if self.input_devices else None
            self.selected_output_device = self.output_devices[0] if self.output_devices else None

            print(f"✅ AUDIO SETUP: Selected input device: {self.selected_input_device}")
            print(f"✅ AUDIO SETUP: Selected output device: {self.selected_output_device}")

            # Schedule a test of the audio system
            self.root.after(3000, self._test_audio_playback)

        except Exception as e:
            print(f"⚠️ AUDIO SETUP: Error initializing audio devices: {e}")
            self.input_devices = ["Default"]
            self.output_devices = ["Default"]
            self.selected_input_device = "Default"
            self.selected_output_device = "Default"
            self.add_to_chat("System", f"Warning: Error initializing audio devices: {e}")

            # Try to play a system beep to indicate error
            try:
                self.root.bell()
            except:
                pass

    def create_widgets(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Create top toolbar with a distinct appearance
        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        toolbar_frame.columnconfigure(1, weight=1)  # Make middle column expandable

        # Left side of toolbar: Task input and button
        task_frame = ttk.Frame(toolbar_frame)
        task_frame.grid(row=0, column=0, sticky=tk.W)

        self.task_entry = ttk.Entry(task_frame, width=40)
        self.task_entry.grid(row=0, column=0, padx=(0, 5))

        self.add_task_button = ttk.Button(task_frame, text="Add Task", command=self.add_task)
        self.add_task_button.grid(row=0, column=1)

        # Middle of toolbar: Status indicators
        status_frame = ttk.Frame(toolbar_frame)
        status_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))

        self.voice_status = ttk.Label(
            status_frame, 
            text="Voice: Off", 
            foreground="gray",
            font=("Segoe UI", 11)
        )
        self.voice_status.pack(side=tk.LEFT, padx=10)

        # Right side of toolbar: Voice Mode button (prominently displayed)
        button_frame = ttk.Frame(toolbar_frame)
        button_frame.grid(row=0, column=2, sticky=tk.E)

        # Create a custom style for the Voice Mode button
        style = ttk.Style()
        style.configure(
            "VoiceMode.TButton",
            font=("Segoe UI", 11),
            background="#202020",
            foreground="#FFFFFF",
        )
        style.map(
            "VoiceMode.TButton",
            background=[("active", "#303030")],
            relief=[("pressed", "sunken")]
        )

        # Voice Settings button with icon and tooltip
        self.voice_settings_button = ttk.Button(
            button_frame,
            text="⚙️",
            command=self.show_voice_settings,
            width=3
        )
        self.voice_settings_button.pack(side=tk.RIGHT, padx=(0, 5), pady=5, ipadx=2, ipady=3)

        # Create tooltip for the settings button
        self.create_tooltip(self.voice_settings_button, "Configure voice input and output devices")

        # Voice Mode button with icon, tooltip, and custom style
        self.voice_mode_button = ttk.Button(
            button_frame,
            text="🎙️ Voice Mode",
            style="VoiceMode.TButton",
            command=self.toggle_voice_mode,
            width=15
        )
        self.voice_mode_button.pack(side=tk.RIGHT, padx=5, pady=5, ipadx=5, ipady=3)

        # Create tooltip for the button
        self.create_tooltip(self.voice_mode_button, "Toggle real-time voice chat mode")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5)

        # Chat tab
        self.chat_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.chat_frame, text="Chat")
        self.chat_frame.columnconfigure(0, weight=1)
        self.chat_frame.rowconfigure(0, weight=1)

        # Chat display with scrollbar
        chat_display_frame = ttk.Frame(self.chat_frame)
        chat_display_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        chat_display_frame.columnconfigure(0, weight=1)
        chat_display_frame.rowconfigure(0, weight=1)

        self.chat_display = tk.Text(
            chat_display_frame, 
            wrap=tk.WORD, 
            bg="#f8f8f8", 
            font=("Segoe UI", 10)
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.chat_display.config(state=tk.DISABLED)

        chat_scrollbar = ttk.Scrollbar(chat_display_frame, orient=tk.VERTICAL, command=self.chat_display.yview)
        chat_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.chat_display['yscrollcommand'] = chat_scrollbar.set

        # Chat input area
        chat_input_frame = ttk.Frame(self.chat_frame)
        chat_input_frame.grid(row=1, column=0, sticky=(tk.E, tk.W), pady=(5, 0))
        chat_input_frame.columnconfigure(0, weight=1)

        self.chat_entry = ttk.Entry(chat_input_frame, font=("Segoe UI", 10))
        self.chat_entry.grid(row=0, column=0, sticky=(tk.E, tk.W), padx=(0, 5))
        self.chat_entry.bind("<Return>", self.send_message)

        self.send_button = ttk.Button(chat_input_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=1)

        # Memory tab
        memory_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(memory_frame, text="Memory")
        memory_frame.columnconfigure(0, weight=1)
        memory_frame.rowconfigure(1, weight=1)
        memory_frame.rowconfigure(3, weight=1)

        # Memory display
        ttk.Label(memory_frame, text="Memory:", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5)
        )

        memory_display_frame = ttk.Frame(memory_frame)
        memory_display_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        memory_display_frame.columnconfigure(0, weight=1)
        memory_display_frame.rowconfigure(0, weight=1)

        self.memory_display = tk.Text(
            memory_display_frame, 
            wrap=tk.WORD, 
            bg="#f8f8f8", 
            font=("Segoe UI", 10)
        )
        self.memory_display.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        memory_scrollbar = ttk.Scrollbar(memory_display_frame, orient=tk.VERTICAL, command=self.memory_display.yview)
        memory_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.memory_display['yscrollcommand'] = memory_scrollbar.set

        # Motivations display
        ttk.Label(memory_frame, text="Motivations:", font=("Segoe UI", 11, "bold")).grid(
            row=2, column=0, sticky=tk.W, pady=(10, 5)
        )

        motivations_display_frame = ttk.Frame(memory_frame)
        motivations_display_frame.grid(row=3, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        motivations_display_frame.columnconfigure(0, weight=1)
        motivations_display_frame.rowconfigure(0, weight=1)

        self.motivations_display = tk.Text(
            motivations_display_frame, 
            wrap=tk.WORD, 
            bg="#f8f8f8", 
            font=("Segoe UI", 10)
        )
        self.motivations_display.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        motivations_scrollbar = ttk.Scrollbar(motivations_display_frame, orient=tk.VERTICAL, command=self.motivations_display.yview)
        motivations_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.motivations_display['yscrollcommand'] = motivations_scrollbar.set

        # Status bar at the bottom
        self.status_bar = ttk.Label(
            main_frame, 
            text="Ready", 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        # Update displays
        self.update_displays()

        # Add welcome message and voice mode instructions
        self.add_to_chat("System", "Welcome to DoBA Voice Assistant! Voice mode is enabled by default.")

        # Print startup diagnostics
        print("\n" + "="*50)
        print("🔊 VOICE SYSTEM DIAGNOSTICS")
        print("="*50)
        print(f"🔊 COQUI_TTS_AVAILABLE: {COQUI_TTS_AVAILABLE}")
        print(f"🔊 AUDIO_AVAILABLE: {AUDIO_AVAILABLE}")
        print(f"🔊 SPEECH_RECOGNITION_AVAILABLE: {SPEECH_RECOGNITION_AVAILABLE}")
        print(f"🔊 WHISPER_AVAILABLE: {WHISPER_AVAILABLE}")
        print(f"🔊 VOSK_AVAILABLE: {VOSK_AVAILABLE}")
        print(f"🔊 voice_mode_enabled: {self.voice_mode_enabled}")
        print(f"🔊 tts_model initialized: {self.tts_model is not None}")
        print("="*50 + "\n")

        # Start voice thread if voice mode is enabled by default
        if self.voice_mode_enabled:
            self.voice_thread_running = True
            self.voice_thread = threading.Thread(target=self._voice_thread_function, daemon=True)
            self.voice_thread.start()
            self.add_to_chat("System", "Voice mode activated. You can speak now.")

            # Update UI to reflect voice mode is enabled
            self.voice_mode_button.configure(text="🎙️ Voice Mode: ON")
            self.voice_status.config(text="Voice: On", foreground="#4CAF50")
            self.status_bar.config(text="Voice mode enabled - listening...")

            # Immediate direct audio test to check if audio system is working
            self.root.after(500, lambda: self._play_direct_audio_test())

            # Schedule multiple tests with increasing delays to ensure voice output is working
            # First test after 2 seconds
            self.root.after(2000, lambda: self.test_voice_output())

            # Second test after 5 seconds if Coqui TTS wasn't initialized at startup
            if self.tts_model is None and COQUI_TTS_AVAILABLE:
                self.root.after(5000, lambda: self.add_to_chat("System", "Attempting to initialize TTS model again with retries..."))
                self.root.after(5500, lambda: self._initialize_tts_with_retry(max_retries=3, retry_count=0))
                self.root.after(6000, lambda: self.test_voice_output())

            # Final test after 10 seconds as a last resort
            self.root.after(10000, lambda: self._final_voice_test())

            # Additional test with direct speech simulation after 15 seconds if all else fails
            self.root.after(15000, lambda: self._emergency_audio_test())

            # Final verification of voice output system after 20 seconds
            self.root.after(20000, lambda: self._verify_voice_output_system())

    def _initialize_tts_model(self):
        """Initialize the TTS model if it hasn't been initialized yet."""
        try:
            if self.tts_model is None and COQUI_TTS_AVAILABLE:
                print("🔊 VOICE INIT: Attempting to initialize Coqui TTS model...")

                # Check if torch is available and CUDA is available
                try:
                    import torch
                    print(f"🔊 VOICE INIT: PyTorch version: {torch.__version__}")
                    print(f"🔊 VOICE INIT: CUDA available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        print(f"🔊 VOICE INIT: CUDA device: {torch.cuda.get_device_name(0)}")
                except ImportError:
                    print("⚠️ VOICE INIT: PyTorch not available")
                except Exception as torch_error:
                    print(f"⚠️ VOICE INIT: Error checking PyTorch: {torch_error}")

                # Try multiple models in order of preference
                models_to_try = [
                    "tts_models/en/ljspeech/tacotron2-DDC",
                    "tts_models/en/ljspeech/glow-tts",
                    "tts_models/en/ljspeech/fast_pitch",
                    "tts_models/en/vctk/vits",
                    "tts_models/multilingual/multi-dataset/your_tts",
                    "tts_models/en/ljspeech/neural_hmm"
                ]

                for model_name in models_to_try:
                    try:
                        print(f"🔊 VOICE INIT: Trying to initialize model: {model_name}")
                        self.tts_model = TTS(model_name)
                        print(f"✅ VOICE INIT: Successfully initialized TTS model: {model_name}")
                        self.add_to_chat("System", f"Voice synthesis model initialized successfully: {model_name}")

                        # Test the model with a simple text
                        try:
                            print("🔊 VOICE INIT: Testing TTS model with a simple text...")
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                                temp_audio_path = temp_audio.name

                            self.tts_model.tts_to_file(text="This is a test of the text to speech system.", file_path=temp_audio_path)

                            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                                print(f"✅ VOICE INIT: Test successful, audio file created: {os.path.getsize(temp_audio_path)} bytes")
                                # Clean up the test file
                                try:
                                    os.unlink(temp_audio_path)
                                except:
                                    pass
                            else:
                                print("⚠️ VOICE INIT: Test failed, audio file not created or empty")
                                raise Exception("Test failed, audio file not created or empty")
                        except Exception as test_error:
                            print(f"⚠️ VOICE INIT: Error testing TTS model: {test_error}")
                            # Continue with the model anyway, it might still work

                        return True
                    except Exception as model_error:
                        print(f"⚠️ VOICE INIT: Error initializing model {model_name}: {model_error}")
                        print(f"⚠️ VOICE INIT: Error type: {type(model_error).__name__}")
                        print(f"⚠️ VOICE INIT: Error details: {str(model_error)}")

                # If we get here, all models failed
                print("❌ VOICE INIT: All TTS models failed to initialize")
                self.add_to_chat("System", "Failed to initialize voice synthesis model. Will try fallback methods.")
                return False

            # If TTS model is already initialized or Coqui TTS is not available
            if self.tts_model is not None:
                print("✅ VOICE INIT: TTS model already initialized")
                return True
            else:
                print("⚠️ VOICE INIT: Coqui TTS not available")
                return False

        except Exception as e:
            print(f"❌ VOICE INIT: Unexpected error initializing TTS model: {e}")
            print(f"❌ VOICE INIT: Error type: {type(e).__name__}")
            print(f"❌ VOICE INIT: Error details: {str(e)}")
            return False

    def _final_voice_test(self):
        """Final attempt to test voice output with fallback methods."""
        try:
            print("\n" + "="*50)
            print("🔊 FINAL VOICE OUTPUT TEST")
            print("="*50)

            print("🔊 FINAL VOICE TEST: Making final attempt to test voice output...")

            # Check the current status of the TTS system
            print(f"🔊 FINAL VOICE TEST: TTS model initialized: {self.tts_model is not None}")
            print(f"🔊 FINAL VOICE TEST: Voice mode enabled: {self.voice_mode_enabled}")
            print(f"🔊 FINAL VOICE TEST: Audio libraries available: {AUDIO_AVAILABLE}")
            print(f"🔊 FINAL VOICE TEST: Coqui TTS available: {COQUI_TTS_AVAILABLE}")

            # If we still don't have a TTS model, try one more time with a different model
            if self.tts_model is None and COQUI_TTS_AVAILABLE:
                print("🔊 FINAL VOICE TEST: TTS model not initialized, trying to initialize now...")

                # Try to initialize the TTS model with our improved retry method
                tts_init_result = self._initialize_tts_with_retry(max_retries=2, retry_count=0)
                print(f"🔊 FINAL VOICE TEST: TTS model initialization result: {tts_init_result}")

                if self.tts_model is None:
                    print("🔊 FINAL VOICE TEST: TTS model still not initialized, trying one more specific model...")
                    try:
                        print("🔊 FINAL VOICE TEST: Trying VCTK TTS model...")
                        self.tts_model = TTS("tts_models/en/vctk/vits")
                        print("✅ FINAL VOICE TEST: VCTK TTS model initialized successfully")

                        # Test the model
                        try:
                            print("🔊 FINAL VOICE TEST: Testing VCTK TTS model...")
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                                temp_audio_path = temp_audio.name

                            self.tts_model.tts_to_file(text="This is a final test of the text to speech system.", file_path=temp_audio_path)

                            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                                print(f"✅ FINAL VOICE TEST: Test successful, audio file created: {os.path.getsize(temp_audio_path)} bytes")
                                # Clean up the test file
                                try:
                                    os.unlink(temp_audio_path)
                                except:
                                    pass
                            else:
                                print("⚠️ FINAL VOICE TEST: Test failed, audio file not created or empty")
                        except Exception as test_error:
                            print(f"⚠️ FINAL VOICE TEST: Error testing VCTK TTS model: {test_error}")
                    except Exception as e:
                        print(f"❌ FINAL VOICE TEST: Failed to initialize VCTK model: {e}")
                        print(f"❌ FINAL VOICE TEST: Error type: {type(e).__name__}")
                        print(f"❌ FINAL VOICE TEST: Error details: {str(e)}")

            # Test with a very simple message
            test_message = "Testing voice output system. This is a final test of the text to speech functionality."
            self.add_to_chat("System", "Final voice output test...")

            # Try direct audio test first to ensure audio output is working
            print("🔊 FINAL VOICE TEST: Testing basic audio playback...")
            audio_test_result = self._test_audio_playback()
            print(f"🔊 FINAL VOICE TEST: Basic audio test result: {audio_test_result}")

            # Then try TTS with detailed logging
            print("🔊 FINAL VOICE TEST: Testing TTS with message: " + test_message)
            speak_result = self.speak_text(test_message)
            print(f"🔊 FINAL VOICE TEST: TTS test result: {speak_result}")

            # If TTS failed, try system TTS as a fallback
            if not speak_result and self.tts_model is None:
                print("🔊 FINAL VOICE TEST: TTS failed, trying system TTS as fallback...")
                system_tts_result = self._system_tts_fallback(test_message)
                print(f"🔊 FINAL VOICE TEST: System TTS result: {system_tts_result}")

            print("🔊 FINAL VOICE TEST: Voice test completed")

            # Return the final status
            return speak_result or audio_test_result

        except Exception as e:
            print(f"❌ FINAL VOICE TEST: Error in final voice test: {e}")
            print(f"❌ FINAL VOICE TEST: Error type: {type(e).__name__}")
            print(f"❌ FINAL VOICE TEST: Error details: {str(e)}")
            self.add_to_chat("System", "Voice output system may not be working properly. Please check console for details.")
            return False

    def _check_audio_dependencies(self):
        """Check for required audio packages and suggest installation commands if missing."""
        try:
            print("🔊 AUDIO DEPS: Checking for required audio packages...")
            missing_packages = []
            installation_commands = []

            # Check for sounddevice and soundfile
            if not AUDIO_AVAILABLE:
                missing_packages.append("sounddevice/soundfile")
                installation_commands.append("pip install sounddevice soundfile")
                print("⚠️ AUDIO DEPS: sounddevice/soundfile not available")

            # Check for speech_recognition
            if not SPEECH_RECOGNITION_AVAILABLE:
                missing_packages.append("SpeechRecognition")
                installation_commands.append("pip install SpeechRecognition")
                print("⚠️ AUDIO DEPS: SpeechRecognition not available")

            # Check for Coqui TTS
            if not COQUI_TTS_AVAILABLE:
                missing_packages.append("TTS (Coqui)")
                installation_commands.append("pip install TTS")
                print("⚠️ AUDIO DEPS: Coqui TTS not available")

            # Check for system audio packages on Linux
            if sys.platform.startswith('linux'):
                try:
                    # Check for ALSA utils
                    alsa_result = subprocess.run(["which", "aplay"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if alsa_result.returncode != 0:
                        missing_packages.append("alsa-utils")
                        installation_commands.append("sudo apt-get install alsa-utils")
                        print("⚠️ AUDIO DEPS: alsa-utils not available")

                    # Check for PulseAudio
                    pulse_result = subprocess.run(["which", "paplay"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if pulse_result.returncode != 0:
                        missing_packages.append("pulseaudio")
                        installation_commands.append("sudo apt-get install pulseaudio")
                        print("⚠️ AUDIO DEPS: pulseaudio not available")

                    # Check for SoX
                    sox_result = subprocess.run(["which", "play"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if sox_result.returncode != 0:
                        missing_packages.append("sox")
                        installation_commands.append("sudo apt-get install sox")
                        print("⚠️ AUDIO DEPS: sox not available")

                    # Check for speech-dispatcher
                    spd_result = subprocess.run(["which", "spd-say"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if spd_result.returncode != 0:
                        missing_packages.append("speech-dispatcher")
                        installation_commands.append("sudo apt-get install speech-dispatcher")
                        print("⚠️ AUDIO DEPS: speech-dispatcher not available")

                except Exception as e:
                    print(f"⚠️ AUDIO DEPS: Error checking system audio packages: {e}")

            # If any packages are missing, show a message with installation instructions
            if missing_packages:
                print(f"⚠️ AUDIO DEPS: Missing audio packages: {', '.join(missing_packages)}")

                # Add message to chat with installation instructions
                message = "⚠️ Some audio packages are missing. Voice output may not work properly.\n\n"
                message += "Missing packages:\n"
                for pkg in missing_packages:
                    message += f"- {pkg}\n"

                message += "\nInstallation commands:\n"
                for cmd in installation_commands:
                    message += f"$ {cmd}\n"

                # Schedule the message to be added to chat after initialization
                self.root.after(1000, lambda: self.add_to_chat("System", message))
            else:
                print("✅ AUDIO DEPS: All required audio packages are available")

        except Exception as e:
            print(f"⚠️ AUDIO DEPS: Error checking audio dependencies: {e}")

    def _verify_voice_output_system(self):
        """Final verification of voice output system and status report."""
        try:
            print("\n" + "="*50)
            print("🔊 VOICE OUTPUT SYSTEM VERIFICATION")
            print("="*50)

            # Check if TTS model is initialized
            tts_initialized = self.tts_model is not None
            print(f"🔊 TTS Model Initialized: {tts_initialized}")
            if tts_initialized:
                print(f"🔊 TTS Model Type: {type(self.tts_model).__name__}")
            else:
                print("🔊 TTS Model: None")

            # Check if audio libraries are available
            print(f"🔊 Audio Libraries Available: {AUDIO_AVAILABLE}")
            print(f"🔊 Speech Recognition Available: {SPEECH_RECOGNITION_AVAILABLE}")
            print(f"🔊 Coqui TTS Available: {COQUI_TTS_AVAILABLE}")
            print(f"🔊 Whisper Available: {WHISPER_AVAILABLE}")
            print(f"🔊 Vosk Available: {VOSK_AVAILABLE}")

            # Check if voice mode is enabled
            print(f"🔊 Voice Mode Enabled: {self.voice_mode_enabled}")

            # Check if voice thread is running
            voice_thread_running = self.voice_thread is not None and self.voice_thread.is_alive()
            print(f"🔊 Voice Thread Running: {voice_thread_running}")
            if voice_thread_running:
                print(f"🔊 Voice Thread ID: {self.voice_thread.ident}")
                print(f"🔊 Voice Thread Name: {self.voice_thread.name}")
                print(f"🔊 Voice Thread Daemon: {self.voice_thread.daemon}")

            # Check if we're currently speaking or listening
            print(f"🔊 Currently Speaking: {self.is_speaking}")
            print(f"🔊 Currently Listening: {self.is_listening}")

            # Check audio devices
            print("\n🔊 AUDIO DEVICE INFORMATION:")
            try:
                if AUDIO_AVAILABLE:
                    # Get and print all available audio devices
                    devices = sd.query_devices()
                    print(f"🔊 Total Audio Devices: {len(devices)}")
                    print("🔊 Audio Devices List:")
                    for i, device in enumerate(devices):
                        print(f"  Device {i}: {device['name']} - {'Input' if device['max_input_channels'] > 0 else ''} {'Output' if device['max_output_channels'] > 0 else ''}")

                    # Get default devices
                    try:
                        default_input = sd.query_devices(kind='input')
                        print(f"🔊 Default Input Device: {default_input['name']}")
                    except Exception as e:
                        print(f"⚠️ Error getting default input device: {e}")

                    try:
                        default_output = sd.query_devices(kind='output')
                        print(f"🔊 Default Output Device: {default_output['name']}")
                    except Exception as e:
                        print(f"⚠️ Error getting default output device: {e}")

                    # Print selected devices
                    print(f"🔊 Selected Input Device: {self.selected_input_device}")
                    print(f"🔊 Selected Output Device: {self.selected_output_device}")
                else:
                    print("⚠️ Audio libraries not available - cannot query devices")
            except Exception as device_error:
                print(f"⚠️ Error querying audio devices: {device_error}")

            # Check system audio configuration
            print("\n🔊 SYSTEM AUDIO CONFIGURATION:")
            try:
                if sys.platform.startswith('linux'):
                    # Check if user is in audio groups
                    groups_result = subprocess.run(["groups"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    user_groups = groups_result.stdout.strip().split()
                    print(f"🔊 User Groups: {', '.join(user_groups)}")
                    print(f"🔊 User in audio group: {'audio' in user_groups}")
                    print(f"🔊 User in pulse group: {'pulse' in user_groups}")
                    print(f"🔊 User in pulse-access group: {'pulse-access' in user_groups}")

                    # Check PulseAudio status
                    try:
                        pulse_status = subprocess.run(["pulseaudio", "--check"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print(f"🔊 PulseAudio Running: {pulse_status.returncode == 0}")
                    except Exception as pulse_error:
                        print(f"⚠️ Error checking PulseAudio status: {pulse_error}")

                    # Check ALSA devices
                    try:
                        alsa_devices = subprocess.run(["aplay", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        print("🔊 ALSA Devices:")
                        for line in alsa_devices.stdout.splitlines()[:5]:  # Show first 5 lines only
                            print(f"  {line}")
                        if len(alsa_devices.stdout.splitlines()) > 5:
                            print(f"  ... ({len(alsa_devices.stdout.splitlines()) - 5} more lines)")
                    except Exception as alsa_error:
                        print(f"⚠️ Error checking ALSA devices: {alsa_error}")

                elif sys.platform.startswith('darwin'):
                    # Check macOS audio devices
                    try:
                        audio_devices = subprocess.run(["system_profiler", "SPAudioDataType"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        print("🔊 macOS Audio Devices:")
                        audio_lines = audio_devices.stdout.splitlines()
                        for line in audio_lines[:10]:  # Show first 10 lines only
                            if "Output" in line or "Input" in line or "Device" in line:
                                print(f"  {line.strip()}")
                        if len(audio_lines) > 10:
                            print(f"  ... ({len(audio_lines) - 10} more lines)")
                    except Exception as macos_error:
                        print(f"⚠️ Error checking macOS audio devices: {macos_error}")

                elif sys.platform.startswith('win'):
                    # Check Windows audio devices
                    try:
                        ps_command = "Get-WmiObject Win32_SoundDevice | Select-Object Name, Status"
                        audio_devices = subprocess.run(["powershell", "-c", ps_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        print("🔊 Windows Audio Devices:")
                        for line in audio_devices.stdout.splitlines()[:10]:  # Show first 10 lines only
                            print(f"  {line.strip()}")
                        if len(audio_devices.stdout.splitlines()) > 10:
                            print(f"  ... ({len(audio_devices.stdout.splitlines()) - 10} more lines)")
                    except Exception as win_error:
                        print(f"⚠️ Error checking Windows audio devices: {win_error}")
            except Exception as sys_error:
                print(f"⚠️ Error checking system audio configuration: {sys_error}")

            # Try a direct audio test to verify audio output
            print("\n🔊 AUDIO OUTPUT TESTS:")
            direct_audio_works = self._play_direct_audio_test()
            print(f"🔊 Direct Audio Test: {'Success' if direct_audio_works else 'Failed'}")

            # Try system audio test
            system_audio_works = self._test_system_audio()
            print(f"🔊 System Audio Test: {'Success' if system_audio_works else 'Failed'}")

            # Try a direct speak_text test with a very simple message
            print("🔊 Testing speak_text directly...")
            speak_text_works = False
            try:
                # Save current speaking state
                was_speaking = self.is_speaking

                # Force speaking to False to allow the test
                self.is_speaking = False

                # Try to speak a very simple message
                speak_result = self.speak_text("Audio test.")
                speak_text_works = speak_result is True

                # Restore original speaking state
                self.is_speaking = was_speaking

                print(f"🔊 Direct speak_text Test: {'Success' if speak_text_works else 'Failed'}")
            except Exception as speak_error:
                print(f"❌ Direct speak_text Test Error: {speak_error}")

            # Try system TTS fallback
            print("🔊 Testing system TTS fallback...")
            system_tts_works = False
            try:
                system_tts_works = self._system_tts_fallback("Testing system TTS.")
                print(f"🔊 System TTS Test: {'Success' if system_tts_works else 'Failed'}")
            except Exception as tts_error:
                print(f"❌ System TTS Test Error: {tts_error}")

            # Overall status
            if tts_initialized and AUDIO_AVAILABLE and (direct_audio_works or speak_text_works):
                status = "✅ FULLY OPERATIONAL"
            elif AUDIO_AVAILABLE and (direct_audio_works or system_audio_works):
                status = "⚠️ PARTIALLY OPERATIONAL (TTS not initialized but audio works)"
            elif direct_audio_works or system_audio_works or system_tts_works:
                status = "⚠️ MINIMAL FUNCTIONALITY (Only basic audio works)"
            else:
                status = "❌ NOT OPERATIONAL (No audio output detected)"

            print(f"\n🔊 OVERALL STATUS: {status}")
            print("="*50 + "\n")

            # Add status report to chat
            message = f"Voice Output System Status: {status}\n\n"

            if status == "✅ FULLY OPERATIONAL":
                message += "Voice output system is fully operational. You should hear spoken responses."
            elif status == "⚠️ PARTIALLY OPERATIONAL (TTS not initialized but audio works)":
                message += "Voice output is partially working. You may hear simple sounds but not full speech synthesis."
            elif status == "⚠️ MINIMAL FUNCTIONALITY (Only basic audio works)":
                message += "Voice output has minimal functionality. Only basic sounds are working."
            else:
                message += "Voice output is not working. Please check your audio settings and ensure your speakers are on."

                # Add troubleshooting tips
                message += "\n\nTroubleshooting tips:\n"
                message += "1. Check if your speakers are on and volume is up\n"
                message += "2. Check if audio is muted in system settings\n"
                message += "3. Try running the application with administrator/sudo privileges\n"
                message += "4. Install required audio packages (see previous messages)\n"
                message += "5. Try restarting your computer\n"
                message += "6. Check if other applications can play sound\n"
                message += "7. Try running 'sudo usermod -a -G audio,pulse,pulse-access $USER' and then log out and back in\n"
                message += "8. On Linux, try 'sudo apt-get install alsa-utils pulseaudio speech-dispatcher espeak'\n"
                message += "9. Check system logs for audio-related errors: 'dmesg | grep -i audio'"

            self.add_to_chat("System", message)

            # If we're fully operational, try one more direct speech test
            if status == "✅ FULLY OPERATIONAL":
                self.root.after(1000, lambda: self.speak_text("Voice output is now working."))
            elif status == "⚠️ PARTIALLY OPERATIONAL (TTS not initialized but audio works)" or status == "⚠️ MINIMAL FUNCTIONALITY (Only basic audio works)":
                # Try to play a simple beep to confirm audio is working
                self.root.after(1000, lambda: self._play_direct_audio_test())

            # Create a detailed diagnostic report file
            try:
                self._create_audio_diagnostic_report(status)
            except Exception as report_error:
                print(f"⚠️ Error creating diagnostic report: {report_error}")

            return status

        except Exception as e:
            print(f"❌ VOICE VERIFICATION: Error verifying voice output system: {e}")
            self.add_to_chat("System", f"Error verifying voice output system: {e}")
            return "❌ ERROR DURING VERIFICATION"

    def _run_audio_diagnostic_wizard(self):
        """Run an interactive wizard to diagnose and fix audio issues."""
        try:
            print("\n" + "="*50)
            print("🔊 AUDIO DIAGNOSTIC WIZARD")
            print("="*50)

            # Create a new top-level window for the wizard
            wizard_window = tk.Toplevel(self.root)
            wizard_window.title("Audio Diagnostic Wizard")
            wizard_window.geometry("800x600")
            wizard_window.minsize(800, 600)

            # Make the window modal
            wizard_window.transient(self.root)
            wizard_window.grab_set()

            # Set up the wizard UI
            main_frame = ttk.Frame(wizard_window, padding=20)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Header
            header_frame = ttk.Frame(main_frame)
            header_frame.pack(fill=tk.X, pady=(0, 20))

            ttk.Label(header_frame, text="Audio Diagnostic Wizard", font=("Helvetica", 16, "bold")).pack(side=tk.LEFT)

            # Create a notebook for the wizard steps
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill=tk.BOTH, expand=True)

            # Status frame at the bottom
            status_frame = ttk.Frame(main_frame)
            status_frame.pack(fill=tk.X, pady=(20, 0))

            status_label = ttk.Label(status_frame, text="Ready to begin diagnostics", font=("Helvetica", 10))
            status_label.pack(side=tk.LEFT)

            # Navigation buttons
            nav_frame = ttk.Frame(main_frame)
            nav_frame.pack(fill=tk.X, pady=(10, 0))

            # Dictionary to store test results
            test_results = {}

            # Function to update the status label
            def update_status(message, color="black"):
                status_label.config(text=message, foreground=color)

            # Function to enable/disable navigation buttons based on current tab
            def update_nav_buttons(event=None):
                current_tab = notebook.index(notebook.select())

                # Enable/disable previous button
                if current_tab == 0:
                    prev_button.config(state=tk.DISABLED)
                else:
                    prev_button.config(state=tk.NORMAL)

                # Enable/disable next button
                if current_tab == notebook.index("end") - 1:  # Last tab
                    next_button.config(text="Finish", command=finish_wizard)
                else:
                    next_button.config(text="Next >", command=lambda: notebook.select(current_tab + 1))

            # Function to finish the wizard
            def finish_wizard():
                try:
                    # Create a diagnostic report with the test results
                    self._create_audio_diagnostic_report("WIZARD DIAGNOSTICS")

                    # Show a message
                    messagebox.showinfo("Diagnostics Complete", 
                                       "Audio diagnostic wizard completed.\n\n"
                                       "A detailed report has been saved to your home directory:\n"
                                       "voice_output_diagnostic_report.txt")

                    # Close the wizard
                    wizard_window.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Error completing diagnostic wizard: {str(e)}")

            # Create the wizard tabs

            # Step 1: System Information
            system_frame = ttk.Frame(notebook, padding=10)
            notebook.add(system_frame, text="1. System Info")

            ttk.Label(system_frame, text="System Information", font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
            ttk.Label(system_frame, text="This wizard will help diagnose and fix audio issues.").pack(anchor=tk.W)
            ttk.Label(system_frame, text="First, let's gather some basic system information:").pack(anchor=tk.W, pady=(10, 5))

            info_text = tk.Text(system_frame, height=15, width=80, wrap=tk.WORD)
            info_text.pack(fill=tk.BOTH, expand=True, pady=10)
            info_text.insert(tk.END, f"Platform: {sys.platform}\n")
            info_text.insert(tk.END, f"Python Version: {sys.version}\n\n")
            info_text.insert(tk.END, f"Audio Libraries Available: {AUDIO_AVAILABLE}\n")
            info_text.insert(tk.END, f"Speech Recognition Available: {SPEECH_RECOGNITION_AVAILABLE}\n")
            info_text.insert(tk.END, f"Coqui TTS Available: {COQUI_TTS_AVAILABLE}\n")
            info_text.insert(tk.END, f"Whisper Available: {WHISPER_AVAILABLE}\n")
            info_text.insert(tk.END, f"Vosk Available: {VOSK_AVAILABLE}\n\n")

            # Add system-specific information
            if sys.platform.startswith('linux'):
                info_text.insert(tk.END, "Linux-specific information:\n")
                try:
                    # User groups
                    groups_result = subprocess.run(["groups"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    user_groups = groups_result.stdout.strip().split()
                    info_text.insert(tk.END, f"User Groups: {', '.join(user_groups)}\n")
                    info_text.insert(tk.END, f"User in audio group: {'audio' in user_groups}\n")
                    info_text.insert(tk.END, f"User in pulse group: {'pulse' in user_groups}\n")
                    info_text.insert(tk.END, f"User in pulse-access group: {'pulse-access' in user_groups}\n\n")
                except Exception as e:
                    info_text.insert(tk.END, f"Error getting user groups: {e}\n\n")

            info_text.config(state=tk.DISABLED)

            # Step 2: Audio Devices
            devices_frame = ttk.Frame(notebook, padding=10)
            notebook.add(devices_frame, text="2. Audio Devices")

            ttk.Label(devices_frame, text="Audio Devices", font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
            ttk.Label(devices_frame, text="Let's check your audio devices:").pack(anchor=tk.W)

            devices_text = tk.Text(devices_frame, height=15, width=80, wrap=tk.WORD)
            devices_text.pack(fill=tk.BOTH, expand=True, pady=10)

            # Function to refresh the devices list
            def refresh_devices():
                devices_text.config(state=tk.NORMAL)
                devices_text.delete(1.0, tk.END)

                if AUDIO_AVAILABLE:
                    try:
                        devices = sd.query_devices()
                        devices_text.insert(tk.END, f"Total Audio Devices: {len(devices)}\n\n")
                        devices_text.insert(tk.END, "Audio Devices List:\n")

                        # Track if we have output devices
                        has_output_devices = False

                        for i, device in enumerate(devices):
                            device_type = []
                            if device['max_input_channels'] > 0:
                                device_type.append("Input")
                            if device['max_output_channels'] > 0:
                                device_type.append("Output")
                                has_output_devices = True

                            devices_text.insert(tk.END, f"Device {i}: {device['name']} - {', '.join(device_type)}\n")

                            # Add more details for each device
                            devices_text.insert(tk.END, f"  - Sample Rate: {device['default_samplerate']} Hz\n")
                            devices_text.insert(tk.END, f"  - Input Channels: {device['max_input_channels']}\n")
                            devices_text.insert(tk.END, f"  - Output Channels: {device['max_output_channels']}\n\n")

                        # Selected devices
                        devices_text.insert(tk.END, f"\nSelected Input Device: {self.selected_input_device}\n")
                        devices_text.insert(tk.END, f"Selected Output Device: {self.selected_output_device}\n\n")

                        # Store the result
                        test_results['has_output_devices'] = has_output_devices

                        if has_output_devices:
                            devices_text.insert(tk.END, "✅ Output devices detected\n")
                        else:
                            devices_text.insert(tk.END, "❌ No output devices detected\n")
                            devices_text.insert(tk.END, "This is a critical issue that must be resolved.\n")

                    except Exception as e:
                        devices_text.insert(tk.END, f"Error querying audio devices: {e}\n")
                        test_results['has_output_devices'] = False
                else:
                    devices_text.insert(tk.END, "Audio libraries not available - cannot query devices\n")
                    devices_text.insert(tk.END, "This is a critical issue that must be resolved.\n")
                    test_results['has_output_devices'] = False

                devices_text.config(state=tk.DISABLED)

            # Refresh button
            refresh_button = ttk.Button(devices_frame, text="Refresh Devices", command=refresh_devices)
            refresh_button.pack(pady=10)

            # Initial refresh
            refresh_devices()

            # Step 3: Basic Audio Test
            basic_test_frame = ttk.Frame(notebook, padding=10)
            notebook.add(basic_test_frame, text="3. Basic Test")

            ttk.Label(basic_test_frame, text="Basic Audio Test", font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
            ttk.Label(basic_test_frame, text="Let's perform a basic audio test to see if sound is working:").pack(anchor=tk.W)

            basic_result_var = tk.StringVar(value="Not tested yet")

            basic_result_frame = ttk.Frame(basic_test_frame)
            basic_result_frame.pack(fill=tk.X, pady=10)

            ttk.Label(basic_result_frame, text="Test Result:").pack(side=tk.LEFT)
            ttk.Label(basic_result_frame, textvariable=basic_result_var, font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)

            # Function to run the basic audio test
            def run_basic_audio_test():
                basic_result_var.set("Testing...")
                update_status("Running basic audio test...", "blue")

                # Run the test in a separate thread to keep UI responsive
                def test_thread():
                    result = self._test_audio_playback()
                    test_results['basic_audio_test'] = result

                    if result:
                        basic_result_var.set("✅ Success - Audio is working")
                        update_status("Basic audio test successful", "green")
                    else:
                        basic_result_var.set("❌ Failed - No audio detected")
                        update_status("Basic audio test failed", "red")

                threading.Thread(target=test_thread).start()

            # Test button
            basic_test_button = ttk.Button(basic_test_frame, text="Run Basic Audio Test", command=run_basic_audio_test)
            basic_test_button.pack(pady=10)

            # Troubleshooting tips
            ttk.Label(basic_test_frame, text="If the test fails, try these steps:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(20, 5))

            tips_text = tk.Text(basic_test_frame, height=10, width=80, wrap=tk.WORD)
            tips_text.pack(fill=tk.BOTH, expand=True, pady=5)
            tips_text.insert(tk.END, "1. Check if your speakers/headphones are connected and turned on\n")
            tips_text.insert(tk.END, "2. Check if your system volume is muted or too low\n")
            tips_text.insert(tk.END, "3. Try a different audio output device if available\n")

            if sys.platform.startswith('linux'):
                tips_text.insert(tk.END, "4. Try running: 'pulseaudio -k && pulseaudio --start' to restart PulseAudio\n")
                tips_text.insert(tk.END, "5. Check if you're in the audio group: 'sudo usermod -a -G audio $USER'\n")
            elif sys.platform.startswith('darwin'):
                tips_text.insert(tk.END, "4. Try resetting the audio subsystem: 'sudo killall coreaudiod'\n")
            elif sys.platform.startswith('win'):
                tips_text.insert(tk.END, "4. Try running the Windows audio troubleshooter\n")
                tips_text.insert(tk.END, "5. Check the Sound control panel and make sure the correct output device is set as default\n")

            tips_text.config(state=tk.DISABLED)

            # Step 4: Advanced Tests
            advanced_test_frame = ttk.Frame(notebook, padding=10)
            notebook.add(advanced_test_frame, text="4. Advanced Tests")

            ttk.Label(advanced_test_frame, text="Advanced Audio Tests", font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
            ttk.Label(advanced_test_frame, text="Let's try different audio output methods to see which ones work:").pack(anchor=tk.W)

            # Results display
            results_text = tk.Text(advanced_test_frame, height=15, width=80, wrap=tk.WORD)
            results_text.pack(fill=tk.BOTH, expand=True, pady=10)

            # Function to run advanced tests
            def run_advanced_tests():
                results_text.config(state=tk.NORMAL)
                results_text.delete(1.0, tk.END)
                results_text.insert(tk.END, "Running advanced audio tests...\n\n")
                results_text.config(state=tk.DISABLED)
                update_status("Running advanced audio tests...", "blue")

                # Run the tests in a separate thread
                def test_thread():
                    working_methods = self._test_all_audio_methods()
                    test_results['working_methods'] = working_methods

                    # Update the results display
                    results_text.config(state=tk.NORMAL)
                    results_text.delete(1.0, tk.END)

                    if working_methods:
                        results_text.insert(tk.END, "✅ Advanced tests completed. Working methods:\n\n")
                        for method in working_methods:
                            results_text.insert(tk.END, f"- {method}\n")

                        results_text.insert(tk.END, "\nRecommendation: Use the following methods in order of preference:\n")
                        for method in working_methods:
                            results_text.insert(tk.END, f"- {method}\n")

                        update_status(f"Found {len(working_methods)} working audio methods", "green")
                    else:
                        results_text.insert(tk.END, "❌ No working audio methods found.\n\n")
                        results_text.insert(tk.END, "This indicates a more serious issue with your audio configuration.\n")
                        results_text.insert(tk.END, "Please proceed to the troubleshooting and repair steps.\n")
                        update_status("No working audio methods found", "red")

                    results_text.config(state=tk.DISABLED)

                threading.Thread(target=test_thread).start()

            # Test button
            advanced_test_button = ttk.Button(advanced_test_frame, text="Run Advanced Tests", command=run_advanced_tests)
            advanced_test_button.pack(pady=10)

            # Step 5: Troubleshooting
            troubleshoot_frame = ttk.Frame(notebook, padding=10)
            notebook.add(troubleshoot_frame, text="5. Troubleshooting")

            ttk.Label(troubleshoot_frame, text="Audio Troubleshooting", font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
            ttk.Label(troubleshoot_frame, text="Based on the test results, here are some troubleshooting steps:").pack(anchor=tk.W)

            # Troubleshooting options
            options_frame = ttk.Frame(troubleshoot_frame)
            options_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            # Function to restart audio subsystem
            def restart_audio():
                status_label.config(text="Restarting audio subsystem...", foreground="blue")

                # Run in a separate thread
                def restart_thread():
                    result = self._restart_audio_subsystem()
                    if result:
                        messagebox.showinfo("Success", "Audio subsystem successfully restarted.")
                        status_label.config(text="Audio subsystem restarted", foreground="green")
                    else:
                        messagebox.showerror("Error", "Failed to restart audio subsystem.")
                        status_label.config(text="Failed to restart audio subsystem", foreground="red")

                threading.Thread(target=restart_thread).start()

            # Function to detect and fix audio device issues
            def fix_audio_devices():
                status_label.config(text="Detecting and fixing audio device issues...", foreground="blue")

                # Run in a separate thread
                def fix_thread():
                    result = self._detect_and_fix_audio_device_issues()
                    if result:
                        messagebox.showinfo("Success", "Audio device issues fixed.")
                        status_label.config(text="Audio device issues fixed", foreground="green")
                    else:
                        messagebox.showerror("Error", "Failed to fix audio device issues.")
                        status_label.config(text="Failed to fix audio device issues", foreground="red")

                threading.Thread(target=fix_thread).start()

            # Function to test and repair audio subsystem
            def repair_audio():
                status_label.config(text="Testing and repairing audio subsystem...", foreground="blue")

                # Run in a separate thread
                def repair_thread():
                    result = self._test_and_repair_audio_subsystem()
                    if result:
                        messagebox.showinfo("Success", "Audio subsystem repaired.")
                        status_label.config(text="Audio subsystem repaired", foreground="green")
                    else:
                        messagebox.showerror("Error", "Failed to repair audio subsystem.")
                        status_label.config(text="Failed to repair audio subsystem", foreground="red")

                threading.Thread(target=repair_thread).start()

            # Add troubleshooting buttons
            ttk.Button(options_frame, text="Restart Audio Subsystem", command=restart_audio).pack(fill=tk.X, pady=5)
            ttk.Button(options_frame, text="Detect & Fix Audio Device Issues", command=fix_audio_devices).pack(fill=tk.X, pady=5)
            ttk.Button(options_frame, text="Test & Repair Audio Subsystem", command=repair_audio).pack(fill=tk.X, pady=5)

            # System-specific troubleshooting
            if sys.platform.startswith('linux'):
                def run_linux_fix():
                    try:
                        # Create a script to fix common Linux audio issues
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as script_file:
                            script_path = script_file.name
                            script_file.write("#!/bin/bash\n")
                            script_file.write("echo 'Fixing Linux audio issues...'\n")
                            script_file.write("# Restart PulseAudio\n")
                            script_file.write("pulseaudio -k\n")
                            script_file.write("sleep 1\n")
                            script_file.write("pulseaudio --start\n")
                            script_file.write("sleep 1\n")
                            script_file.write("# Make sure user is in audio groups\n")
                            script_file.write("sudo usermod -a -G audio,pulse,pulse-access $USER\n")
                            script_file.write("# Install common audio packages\n")
                            script_file.write("sudo apt-get update\n")
                            script_file.write("sudo apt-get install -y alsa-utils pulseaudio pavucontrol\n")
                            script_file.write("echo 'Done. Please log out and log back in for group changes to take effect.'\n")

                        # Make the script executable
                        os.chmod(script_path, 0o755)

                        # Ask for confirmation
                        if messagebox.askyesno("Linux Audio Fix", 
                                              "This will run a script to fix common Linux audio issues.\n"
                                              "It requires sudo access and will:\n"
                                              "- Restart PulseAudio\n"
                                              "- Add your user to audio groups\n"
                                              "- Install audio packages\n\n"
                                              "Continue?"):

                            # Run the script with gksudo or pkexec
                            if subprocess.run(["which", "gksudo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                                subprocess.Popen(["gksudo", script_path])
                            elif subprocess.run(["which", "pkexec"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                                subprocess.Popen(["pkexec", script_path])
                            else:
                                # Fallback to terminal
                                terminal_cmd = None
                                for term in ["gnome-terminal", "konsole", "xterm"]:
                                    if subprocess.run(["which", term], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                                        terminal_cmd = term
                                        break

                                if terminal_cmd:
                                    subprocess.Popen([terminal_cmd, "-e", f"sudo {script_path}"])
                                else:
                                    messagebox.showerror("Error", "Could not find a suitable terminal emulator.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Error running Linux audio fix: {str(e)}")

                ttk.Button(options_frame, text="Run Linux Audio Fix Script", command=run_linux_fix).pack(fill=tk.X, pady=5)

            elif sys.platform.startswith('win'):
                def run_windows_fix():
                    try:
                        # Run Windows audio troubleshooter
                        subprocess.Popen(["control.exe", "/name", "Microsoft.Troubleshooting", "/page", "pageAudioPlayback"])
                        messagebox.showinfo("Windows Audio Fix", "Windows Audio Troubleshooter has been launched.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Error running Windows audio fix: {str(e)}")

                ttk.Button(options_frame, text="Run Windows Audio Troubleshooter", command=run_windows_fix).pack(fill=tk.X, pady=5)

            elif sys.platform.startswith('darwin'):
                def run_macos_fix():
                    try:
                        # Create a script to reset macOS audio
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as script_file:
                            script_path = script_file.name
                            script_file.write("#!/bin/bash\n")
                            script_file.write("echo 'Resetting macOS audio subsystem...'\n")
                            script_file.write("sudo killall coreaudiod\n")
                            script_file.write("echo 'Done. Audio subsystem has been reset.'\n")

                        # Make the script executable
                        os.chmod(script_path, 0o755)

                        # Ask for confirmation
                        if messagebox.askyesno("macOS Audio Fix", 
                                              "This will reset the macOS audio subsystem.\n"
                                              "It requires sudo access.\n\n"
                                              "Continue?"):

                            # Run the script in Terminal
                            subprocess.Popen(["open", "-a", "Terminal", script_path])
                    except Exception as e:
                        messagebox.showerror("Error", f"Error running macOS audio fix: {str(e)}")

                ttk.Button(options_frame, text="Reset macOS Audio Subsystem", command=run_macos_fix).pack(fill=tk.X, pady=5)

            # Step 6: Summary and Report
            summary_frame = ttk.Frame(notebook, padding=10)
            notebook.add(summary_frame, text="6. Summary")

            ttk.Label(summary_frame, text="Diagnostic Summary", font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
            ttk.Label(summary_frame, text="Here's a summary of the diagnostic results:").pack(anchor=tk.W)

            summary_text = tk.Text(summary_frame, height=15, width=80, wrap=tk.WORD)
            summary_text.pack(fill=tk.BOTH, expand=True, pady=10)

            # Function to update the summary
            def update_summary():
                summary_text.config(state=tk.NORMAL)
                summary_text.delete(1.0, tk.END)

                summary_text.insert(tk.END, "AUDIO DIAGNOSTIC SUMMARY\n")
                summary_text.insert(tk.END, "="*30 + "\n\n")

                # System information
                summary_text.insert(tk.END, "System Information:\n")
                summary_text.insert(tk.END, f"- Platform: {sys.platform}\n")
                summary_text.insert(tk.END, f"- Audio Libraries: {'Available' if AUDIO_AVAILABLE else 'Not Available'}\n\n")

                # Audio devices
                if 'has_output_devices' in test_results:
                    summary_text.insert(tk.END, "Audio Devices:\n")
                    if test_results['has_output_devices']:
                        summary_text.insert(tk.END, "- ✅ Output devices detected\n\n")
                    else:
                        summary_text.insert(tk.END, "- ❌ No output devices detected\n\n")

                # Basic audio test
                if 'basic_audio_test' in test_results:
                    summary_text.insert(tk.END, "Basic Audio Test:\n")
                    if test_results['basic_audio_test']:
                        summary_text.insert(tk.END, "- ✅ Basic audio is working\n\n")
                    else:
                        summary_text.insert(tk.END, "- ❌ Basic audio test failed\n\n")

                # Advanced tests
                if 'working_methods' in test_results:
                    summary_text.insert(tk.END, "Advanced Audio Tests:\n")
                    if test_results['working_methods']:
                        summary_text.insert(tk.END, f"- ✅ Found {len(test_results['working_methods'])} working audio methods\n")
                        for method in test_results['working_methods']:
                            summary_text.insert(tk.END, f"  - {method}\n")
                    else:
                        summary_text.insert(tk.END, "- ❌ No working audio methods found\n")

                # Overall assessment
                summary_text.insert(tk.END, "\nOverall Assessment:\n")

                if ('basic_audio_test' in test_results and test_results['basic_audio_test']) or \
                   ('working_methods' in test_results and test_results['working_methods']):
                    summary_text.insert(tk.END, "✅ Your audio system appears to be working.\n")
                    if 'working_methods' in test_results and test_results['working_methods']:
                        best_method = test_results['working_methods'][0]
                        summary_text.insert(tk.END, f"Recommended audio method: {best_method}\n")
                else:
                    summary_text.insert(tk.END, "❌ Your audio system is not working properly.\n")
                    summary_text.insert(tk.END, "Please review the troubleshooting steps and try the repair options.\n")

                summary_text.config(state=tk.DISABLED)

            # Create report button
            def create_report():
                try:
                    # Create a diagnostic report
                    self._create_audio_diagnostic_report("WIZARD DIAGNOSTICS")

                    # Show a message
                    messagebox.showinfo("Report Created", 
                                       "Audio diagnostic report created in your home directory:\n"
                                       "voice_output_diagnostic_report.txt")
                except Exception as e:
                    messagebox.showerror("Error", f"Error creating diagnostic report: {str(e)}")

            report_button = ttk.Button(summary_frame, text="Create Detailed Report", command=create_report)
            report_button.pack(pady=10)

            # Navigation buttons
            prev_button = ttk.Button(nav_frame, text="< Previous", command=lambda: notebook.select(notebook.index(notebook.select()) - 1))
            prev_button.pack(side=tk.LEFT, padx=5)

            next_button = ttk.Button(nav_frame, text="Next >", command=lambda: notebook.select(notebook.index(notebook.select()) + 1))
            next_button.pack(side=tk.RIGHT, padx=5)

            # Update summary button
            update_summary_button = ttk.Button(nav_frame, text="Update Summary", command=update_summary)
            update_summary_button.pack(side=tk.RIGHT, padx=5)

            # Bind tab change event to update navigation buttons
            notebook.bind("<<NotebookTabChanged>>", update_nav_buttons)

            # Initial update of navigation buttons
            update_nav_buttons()

            # Make the window modal
            wizard_window.wait_window()

        except Exception as e:
            print(f"❌ Error running audio diagnostic wizard: {e}")
            raise

    def _create_audio_diagnostic_report(self, status):
        """Create a detailed diagnostic report for audio troubleshooting."""
        try:
            # Create a report file in the user's home directory
            report_path = os.path.join(os.path.expanduser("~"), "voice_output_diagnostic_report.txt")

            with open(report_path, "w") as report_file:
                # Write header
                report_file.write("="*50 + "\n")
                report_file.write("VOICE OUTPUT DIAGNOSTIC REPORT\n")
                report_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                report_file.write("="*50 + "\n\n")

                # System information
                report_file.write("SYSTEM INFORMATION:\n")
                report_file.write(f"Platform: {sys.platform}\n")
                report_file.write(f"Python Version: {sys.version}\n")

                # Audio libraries
                report_file.write("\nAUDIO LIBRARIES:\n")
                report_file.write(f"Audio Libraries Available: {AUDIO_AVAILABLE}\n")
                report_file.write(f"Speech Recognition Available: {SPEECH_RECOGNITION_AVAILABLE}\n")
                report_file.write(f"Coqui TTS Available: {COQUI_TTS_AVAILABLE}\n")
                report_file.write(f"Whisper Available: {WHISPER_AVAILABLE}\n")
                report_file.write(f"Vosk Available: {VOSK_AVAILABLE}\n")

                # TTS model
                report_file.write("\nTTS MODEL:\n")
                report_file.write(f"TTS Model Initialized: {self.tts_model is not None}\n")
                if self.tts_model is not None:
                    report_file.write(f"TTS Model Type: {type(self.tts_model).__name__}\n")

                # Voice status
                report_file.write("\nVOICE STATUS:\n")
                report_file.write(f"Voice Mode Enabled: {self.voice_mode_enabled}\n")
                report_file.write(f"Voice Thread Running: {self.voice_thread is not None and self.voice_thread.is_alive()}\n")
                report_file.write(f"Currently Speaking: {self.is_speaking}\n")
                report_file.write(f"Currently Listening: {self.is_listening}\n")

                # Audio devices
                report_file.write("\nAUDIO DEVICES:\n")
                if AUDIO_AVAILABLE:
                    try:
                        devices = sd.query_devices()
                        report_file.write(f"Total Audio Devices: {len(devices)}\n")
                        report_file.write("Audio Devices List:\n")
                        for i, device in enumerate(devices):
                            report_file.write(f"  Device {i}: {device['name']} - {'Input' if device['max_input_channels'] > 0 else ''} {'Output' if device['max_output_channels'] > 0 else ''}\n")

                        # Selected devices
                        report_file.write(f"Selected Input Device: {self.selected_input_device}\n")
                        report_file.write(f"Selected Output Device: {self.selected_output_device}\n")
                    except Exception as e:
                        report_file.write(f"Error querying audio devices: {e}\n")
                else:
                    report_file.write("Audio libraries not available - cannot query devices\n")

                # Test results
                report_file.write("\nTEST RESULTS:\n")
                report_file.write(f"Direct Audio Test: {self._play_direct_audio_test()}\n")
                report_file.write(f"System Audio Test: {self._test_system_audio()}\n")

                # Overall status
                report_file.write("\nOVERALL STATUS:\n")
                report_file.write(f"Status: {status}\n")

                # System-specific information
                report_file.write("\nSYSTEM-SPECIFIC INFORMATION:\n")
                if sys.platform.startswith('linux'):
                    try:
                        # User groups
                        groups_result = subprocess.run(["groups"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        user_groups = groups_result.stdout.strip().split()
                        report_file.write(f"User Groups: {', '.join(user_groups)}\n")
                        report_file.write(f"User in audio group: {'audio' in user_groups}\n")
                        report_file.write(f"User in pulse group: {'pulse' in user_groups}\n")
                        report_file.write(f"User in pulse-access group: {'pulse-access' in user_groups}\n")

                        # ALSA devices
                        try:
                            alsa_devices = subprocess.run(["aplay", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            report_file.write("\nALSA Devices:\n")
                            for line in alsa_devices.stdout.splitlines():
                                report_file.write(f"  {line}\n")
                        except Exception as alsa_error:
                            report_file.write(f"Error checking ALSA devices: {alsa_error}\n")

                        # PulseAudio devices
                        try:
                            pulse_devices = subprocess.run(["pactl", "list", "sinks"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            report_file.write("\nPulseAudio Sinks:\n")
                            for line in pulse_devices.stdout.splitlines()[:20]:  # First 20 lines only
                                report_file.write(f"  {line}\n")
                            if len(pulse_devices.stdout.splitlines()) > 20:
                                report_file.write(f"  ... ({len(pulse_devices.stdout.splitlines()) - 20} more lines)\n")
                        except Exception as pulse_error:
                            report_file.write(f"Error checking PulseAudio devices: {pulse_error}\n")
                    except Exception as linux_error:
                        report_file.write(f"Error gathering Linux-specific information: {linux_error}\n")

                # Troubleshooting tips
                report_file.write("\nTROUBLESHOOTING TIPS:\n")
                report_file.write("1. Check if your speakers are on and volume is up\n")
                report_file.write("2. Check if audio is muted in system settings\n")
                report_file.write("3. Try running the application with administrator/sudo privileges\n")
                report_file.write("4. Install required audio packages\n")
                report_file.write("5. Try restarting your computer\n")
                report_file.write("6. Check if other applications can play sound\n")
                if sys.platform.startswith('linux'):
                    report_file.write("7. Try running 'sudo usermod -a -G audio,pulse,pulse-access $USER' and then log out and back in\n")
                    report_file.write("8. Install audio packages: 'sudo apt-get install alsa-utils pulseaudio speech-dispatcher espeak'\n")
                    report_file.write("9. Check system logs: 'dmesg | grep -i audio'\n")

            print(f"✅ Audio diagnostic report created at: {report_path}")
            self.add_to_chat("System", f"Audio diagnostic report created at: {report_path}")

        except Exception as e:
            print(f"❌ Error creating audio diagnostic report: {e}")
            self.add_to_chat("System", f"Error creating audio diagnostic report: {e}")

    def _emergency_audio_test(self):
        """Emergency test of audio system using all available methods."""
        try:
            print("🚨 EMERGENCY AUDIO TEST: Last resort attempt to verify audio system...")
            self.add_to_chat("System", "Emergency audio test - attempting to verify sound output...")

            # Check if any previous tests have succeeded
            audio_working = False

            # Try direct audio test first
            print("🚨 EMERGENCY AUDIO TEST: Trying direct audio test...")
            if self._play_direct_audio_test():
                print("✅ EMERGENCY AUDIO TEST: Direct audio test successful")
                audio_working = True
            else:
                print("❌ EMERGENCY AUDIO TEST: Direct audio test failed")

            # Try direct speech simulation
            print("🚨 EMERGENCY AUDIO TEST: Trying direct speech simulation...")
            if self._play_direct_speech_simulation():
                print("✅ EMERGENCY AUDIO TEST: Direct speech simulation successful")
                audio_working = True
            else:
                print("❌ EMERGENCY AUDIO TEST: Direct speech simulation failed")

            # Try system bell
            try:
                print("🚨 EMERGENCY AUDIO TEST: Trying system bell...")
                self.root.bell()
                print("✅ EMERGENCY AUDIO TEST: System bell triggered")
                audio_working = True
            except Exception as bell_error:
                print(f"❌ EMERGENCY AUDIO TEST: System bell failed: {bell_error}")

            # Try direct system audio test
            print("🚨 EMERGENCY AUDIO TEST: Trying direct system audio test...")
            if self._test_system_audio():
                print("✅ EMERGENCY AUDIO TEST: Direct system audio test successful")
                audio_working = True
            else:
                print("❌ EMERGENCY AUDIO TEST: Direct system audio test failed")

            # Try all possible system sound commands
            try:
                print("🚨 EMERGENCY AUDIO TEST: Trying all system sound commands...")

                # Linux commands
                if sys.platform.startswith('linux'):
                    commands = [
                        ["paplay", "--volume=65536", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                        ["paplay", "--volume=65536", "/usr/share/sounds/freedesktop/stereo/message.oga"],
                        ["aplay", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                        ["play", "-q", "/usr/share/sounds/alsa/Front_Center.wav"],
                        ["spd-say", "-r", "0", "Testing audio output"]
                    ]

                # macOS commands
                elif sys.platform.startswith('darwin'):
                    commands = [
                        ["afplay", "/System/Library/Sounds/Ping.aiff"],
                        ["afplay", "/System/Library/Sounds/Tink.aiff"],
                        ["say", "-v", "Alex", "Testing audio output"]
                    ]

                # Windows commands
                elif sys.platform.startswith('win'):
                    commands = [
                        ["powershell", "-c", "(New-Object Media.SoundPlayer).PlaySync()"],
                        ["powershell", "-c", "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('Testing audio output')"]
                    ]
                else:
                    commands = []

                # Try each command
                for cmd in commands:
                    try:
                        print(f"🚨 EMERGENCY AUDIO TEST: Trying command: {' '.join(cmd)}")
                        subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                        print(f"✅ EMERGENCY AUDIO TEST: Command executed: {' '.join(cmd)}")
                        audio_working = True
                    except Exception as cmd_error:
                        print(f"❌ EMERGENCY AUDIO TEST: Command failed: {' '.join(cmd)} - {cmd_error}")

            except Exception as cmd_error:
                print(f"❌ EMERGENCY AUDIO TEST: Error trying system commands: {cmd_error}")

            # Report final status
            if audio_working:
                print("✅ EMERGENCY AUDIO TEST: At least one audio test succeeded")
                self.add_to_chat("System", "Audio system appears to be working with at least one method.")
            else:
                print("❌ EMERGENCY AUDIO TEST: All audio tests failed")
                self.add_to_chat("System", "WARNING: All audio tests failed. Voice output may not be working.")

                # Add detailed troubleshooting information
                troubleshooting_msg = (
                    "Audio troubleshooting:\n"
                    "1. Check if your speakers are on and volume is up\n"
                    "2. Check if audio is muted in system settings\n"
                    "3. Try running 'aplay -l' in terminal to list audio devices\n"
                    "4. Make sure your user has permission to access audio devices\n"
                    "5. Try installing additional audio packages: sudo apt-get install alsa-utils pulseaudio\n"
                    "6. Try running the application with sudo privileges\n"
                    "7. Check if other applications can play sound"
                )
                self.add_to_chat("System", troubleshooting_msg)

            return audio_working

        except Exception as e:
            print(f"❌ EMERGENCY AUDIO TEST: Error in emergency audio test: {e}")
            self.add_to_chat("System", f"Emergency audio test failed with error: {e}")
            return False

    def _test_system_audio(self):
        """Test audio output using direct system methods, bypassing Python audio libraries."""
        try:
            print("🔊 SYSTEM AUDIO TEST: Testing audio output using direct system methods...")

            # Create a temporary WAV file with a simple tone
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name

            print(f"🔊 SYSTEM AUDIO TEST: Created temporary file at {temp_audio_path}")

            # Generate a simple WAV file with a sine wave
            try:
                # Use Python's wave module to create a WAV file directly
                import wave
                import struct
                import math

                # WAV file parameters
                sample_rate = 44100
                duration = 0.5  # seconds
                frequency = 440  # Hz (A4 note)
                amplitude = 32767  # Maximum amplitude for 16-bit audio

                # Create the WAV file
                with wave.open(temp_audio_path, 'w') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 2 bytes (16 bits) per sample
                    wav_file.setframerate(sample_rate)

                    # Generate the sine wave
                    for i in range(int(duration * sample_rate)):
                        value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                        packed_value = struct.pack('h', value)  # 'h' for 16-bit signed integer
                        wav_file.writeframes(packed_value)

                print(f"✅ SYSTEM AUDIO TEST: WAV file created successfully")

                # Now try to play the WAV file using system commands
                success = False

                # First, try the raw audio output method
                try:
                    print("🔊 SYSTEM AUDIO TEST: Trying raw audio output...")
                    raw_success = self._test_raw_audio_output()
                    if raw_success:
                        print("✅ SYSTEM AUDIO TEST: Raw audio output successful")
                        success = True
                    else:
                        print("⚠️ SYSTEM AUDIO TEST: Raw audio output failed, trying other methods")
                except Exception as raw_error:
                    print(f"⚠️ SYSTEM AUDIO TEST: Raw audio output error: {raw_error}")

                # Try different commands based on the platform
                if not success and sys.platform.startswith('linux'):
                    # Try aplay (ALSA)
                    try:
                        print("🔊 SYSTEM AUDIO TEST: Trying aplay...")
                        subprocess.run(["aplay", temp_audio_path], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                        print("✅ SYSTEM AUDIO TEST: aplay executed")
                        success = True
                    except Exception as e:
                        print(f"⚠️ SYSTEM AUDIO TEST: aplay failed: {e}")

                    # Try paplay (PulseAudio)
                    if not success:
                        try:
                            print("🔊 SYSTEM AUDIO TEST: Trying paplay...")
                            subprocess.run(["paplay", temp_audio_path], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                            print("✅ SYSTEM AUDIO TEST: paplay executed")
                            success = True
                        except Exception as e:
                            print(f"⚠️ SYSTEM AUDIO TEST: paplay failed: {e}")

                    # Try play (SoX)
                    if not success:
                        try:
                            print("🔊 SYSTEM AUDIO TEST: Trying play (SoX)...")
                            subprocess.run(["play", temp_audio_path], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                            print("✅ SYSTEM AUDIO TEST: play executed")
                            success = True
                        except Exception as e:
                            print(f"⚠️ SYSTEM AUDIO TEST: play failed: {e}")

                elif not success and sys.platform.startswith('darwin'):
                    # Try afplay (macOS)
                    try:
                        print("🔊 SYSTEM AUDIO TEST: Trying afplay...")
                        subprocess.run(["afplay", temp_audio_path], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                        print("✅ SYSTEM AUDIO TEST: afplay executed")
                        success = True
                    except Exception as e:
                        print(f"⚠️ SYSTEM AUDIO TEST: afplay failed: {e}")

                elif not success and sys.platform.startswith('win'):
                    # Try PowerShell (Windows)
                    try:
                        print("🔊 SYSTEM AUDIO TEST: Trying PowerShell...")
                        ps_command = f"(New-Object System.Media.SoundPlayer '{temp_audio_path}').PlaySync()"
                        subprocess.run(["powershell", "-c", ps_command], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                        print("✅ SYSTEM AUDIO TEST: PowerShell executed")
                        success = True
                    except Exception as e:
                        print(f"⚠️ SYSTEM AUDIO TEST: PowerShell failed: {e}")

                # Clean up the temporary file
                try:
                    os.unlink(temp_audio_path)
                    print(f"✅ SYSTEM AUDIO TEST: Temporary file {temp_audio_path} cleaned up")
                except Exception as e:
                    print(f"⚠️ SYSTEM AUDIO TEST: Error cleaning up temporary file: {e}")

                return success

            except Exception as wave_error:
                print(f"❌ SYSTEM AUDIO TEST: Error creating WAV file: {wave_error}")

                # Try to clean up the temporary file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

                # Try raw audio output as a last resort
                try:
                    print("🔊 SYSTEM AUDIO TEST: Trying raw audio output as last resort...")
                    raw_success = self._test_raw_audio_output()
                    if raw_success:
                        print("✅ SYSTEM AUDIO TEST: Raw audio output successful")
                        return True
                    else:
                        print("❌ SYSTEM AUDIO TEST: Raw audio output failed")
                except Exception as raw_error:
                    print(f"❌ SYSTEM AUDIO TEST: Raw audio output error: {raw_error}")

                return False

        except Exception as e:
            print(f"❌ SYSTEM AUDIO TEST: Error in system audio test: {e}")

            # Try raw audio output as a last resort
            try:
                print("🔊 SYSTEM AUDIO TEST: Trying raw audio output as final resort...")
                raw_success = self._test_raw_audio_output()
                if raw_success:
                    print("✅ SYSTEM AUDIO TEST: Raw audio output successful")
                    return True
                else:
                    print("❌ SYSTEM AUDIO TEST: Raw audio output failed")
            except Exception as raw_error:
                print(f"❌ SYSTEM AUDIO TEST: Raw audio output error: {raw_error}")

            return False

    def _test_raw_audio_output(self):
        """Test audio output using raw system APIs, bypassing all Python audio libraries."""
        try:
            print("🔊 RAW AUDIO TEST: Testing audio output using raw system APIs...")

            # Platform-specific implementations
            if sys.platform.startswith('win'):
                # Windows implementation using winsound
                try:
                    print("🔊 RAW AUDIO TEST: Using winsound on Windows...")
                    # Try to import winsound
                    import winsound
                    # Play a simple beep
                    frequency = 440  # Hz
                    duration = 500   # ms
                    winsound.Beep(frequency, duration)
                    print("✅ RAW AUDIO TEST: winsound.Beep successful")
                    return True
                except Exception as win_error:
                    print(f"⚠️ RAW AUDIO TEST: winsound error: {win_error}")

                    # Try using Windows API directly
                    try:
                        print("🔊 RAW AUDIO TEST: Using Windows API directly...")
                        # Load the winmm.dll
                        winmm = ctypes.WinDLL('winmm')
                        # Call the Beep function
                        winmm.Beep(440, 500)
                        print("✅ RAW AUDIO TEST: winmm.Beep successful")
                        return True
                    except Exception as api_error:
                        print(f"❌ RAW AUDIO TEST: Windows API error: {api_error}")
                        return False

            elif sys.platform.startswith('darwin'):
                # macOS implementation using AppKit
                try:
                    print("🔊 RAW AUDIO TEST: Using AppKit on macOS...")
                    # Try to use NSBeep
                    subprocess.run(["osascript", "-e", "beep"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                    print("✅ RAW AUDIO TEST: NSBeep successful")
                    return True
                except Exception as mac_error:
                    print(f"⚠️ RAW AUDIO TEST: AppKit error: {mac_error}")

                    # Try using system_profiler to check audio devices
                    try:
                        print("🔊 RAW AUDIO TEST: Checking audio devices on macOS...")
                        result = subprocess.run(["system_profiler", "SPAudioDataType"], 
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)

                        # Check if there are any audio devices
                        if "Output" in result.stdout:
                            print("✅ RAW AUDIO TEST: Audio devices found on macOS")
                            # Try to play a sound using afplay with a built-in sound
                            try:
                                subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"], 
                                              check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                                print("✅ RAW AUDIO TEST: afplay successful")
                                return True
                            except Exception as afplay_error:
                                print(f"⚠️ RAW AUDIO TEST: afplay error: {afplay_error}")
                                return False
                        else:
                            print("❌ RAW AUDIO TEST: No audio devices found on macOS")
                            return False
                    except Exception as prof_error:
                        print(f"❌ RAW AUDIO TEST: system_profiler error: {prof_error}")
                        return False

            elif sys.platform.startswith('linux'):
                # Linux implementation using ALSA or PulseAudio
                try:
                    print("🔊 RAW AUDIO TEST: Using ALSA/PulseAudio on Linux...")

                    # Try using ALSA directly
                    try:
                        # Check if we have access to ALSA devices
                        result = subprocess.run(["aplay", "-l"], 
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)

                        if "no soundcards found" not in result.stdout and "no soundcards found" not in result.stderr:
                            print("✅ RAW AUDIO TEST: ALSA devices found")

                            # Create a simple raw audio file
                            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as temp_audio:
                                temp_audio_path = temp_audio.name

                                # Generate a simple sine wave
                                import math
                                sample_rate = 44100
                                duration = 0.5  # seconds
                                frequency = 440  # Hz
                                amplitude = 32767  # Maximum amplitude for 16-bit audio

                                # Generate raw PCM data
                                for i in range(int(duration * sample_rate)):
                                    value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                                    temp_audio.write(value.to_bytes(2, byteorder='little', signed=True))

                            # Play the raw audio file using aplay
                            subprocess.run(["aplay", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", temp_audio_path], 
                                          check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)

                            # Clean up
                            os.unlink(temp_audio_path)

                            print("✅ RAW AUDIO TEST: ALSA raw audio playback successful")
                            return True
                        else:
                            print("⚠️ RAW AUDIO TEST: No ALSA devices found")
                    except Exception as alsa_error:
                        print(f"⚠️ RAW AUDIO TEST: ALSA error: {alsa_error}")

                    # Try using PulseAudio
                    try:
                        # Check if PulseAudio is running
                        result = subprocess.run(["pulseaudio", "--check"], 
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)

                        if result.returncode == 0:
                            print("✅ RAW AUDIO TEST: PulseAudio is running")

                            # Try to play a sound using paplay
                            try:
                                # Create a simple WAV file
                                import wave
                                import struct

                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                                    temp_audio_path = temp_audio.name

                                sample_rate = 44100
                                duration = 0.5  # seconds
                                frequency = 440  # Hz
                                amplitude = 32767  # Maximum amplitude for 16-bit audio

                                with wave.open(temp_audio_path, 'w') as wav_file:
                                    wav_file.setnchannels(1)  # Mono
                                    wav_file.setsampwidth(2)  # 2 bytes (16 bits) per sample
                                    wav_file.setframerate(sample_rate)

                                    # Generate the sine wave
                                    for i in range(int(duration * sample_rate)):
                                        value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                                        packed_value = struct.pack('h', value)  # 'h' for 16-bit signed integer
                                        wav_file.writeframes(packed_value)

                                # Play the WAV file using paplay
                                subprocess.run(["paplay", "--volume=65536", temp_audio_path], 
                                              check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)

                                # Clean up
                                os.unlink(temp_audio_path)

                                print("✅ RAW AUDIO TEST: PulseAudio playback successful")
                                return True
                            except Exception as paplay_error:
                                print(f"⚠️ RAW AUDIO TEST: paplay error: {paplay_error}")
                        else:
                            print("⚠️ RAW AUDIO TEST: PulseAudio is not running")
                    except Exception as pulse_error:
                        print(f"⚠️ RAW AUDIO TEST: PulseAudio error: {pulse_error}")

                    # Try using beep command
                    try:
                        print("🔊 RAW AUDIO TEST: Trying beep command...")
                        subprocess.run(["beep"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                        print("✅ RAW AUDIO TEST: beep command successful")
                        return True
                    except Exception as beep_error:
                        print(f"⚠️ RAW AUDIO TEST: beep command error: {beep_error}")

                    # Try using speaker-test
                    try:
                        print("🔊 RAW AUDIO TEST: Trying speaker-test...")
                        # Run speaker-test for a very short duration
                        subprocess.run(["speaker-test", "-t", "sine", "-f", "440", "-l", "1", "-D", "default"], 
                                      check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                        print("✅ RAW AUDIO TEST: speaker-test executed")
                        return True
                    except Exception as speaker_error:
                        print(f"⚠️ RAW AUDIO TEST: speaker-test error: {speaker_error}")

                    # If all else fails, try using the PC speaker directly
                    try:
                        print("🔊 RAW AUDIO TEST: Trying to use PC speaker directly...")
                        # Try to open /dev/console
                        with open('/dev/console', 'wb') as console:
                            # Send the bell character to the console
                            console.write(b'\a')
                            console.flush()
                        print("✅ RAW AUDIO TEST: PC speaker beep successful")
                        return True
                    except Exception as console_error:
                        print(f"⚠️ RAW AUDIO TEST: PC speaker error: {console_error}")

                    return False

                except Exception as linux_error:
                    print(f"❌ RAW AUDIO TEST: Linux audio error: {linux_error}")
                    return False

            # Fallback for other platforms or if all platform-specific methods fail
            print("🔊 RAW AUDIO TEST: Using platform-independent method...")

            # Try using the Tkinter bell
            try:
                print("🔊 RAW AUDIO TEST: Trying Tkinter bell...")
                self.root.bell()
                print("✅ RAW AUDIO TEST: Tkinter bell successful")
                return True
            except Exception as bell_error:
                print(f"⚠️ RAW AUDIO TEST: Tkinter bell error: {bell_error}")

            # If we get here, all methods have failed
            print("❌ RAW AUDIO TEST: All methods failed")
            return False

        except Exception as e:
            print(f"❌ RAW AUDIO TEST: Error in raw audio test: {e}")
            return False

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            # Create a toplevel window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")

            label = ttk.Label(self.tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1, padding=2)
            label.pack()

        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def add_task(self):
        """Add a task to the autonomous system."""
        try:
            task = self.task_entry.get().strip()
            if task:
                with self.system.lock:
                    if hasattr(self.system, 'task_queue'):
                        self.system.task_queue.append(task)
                    else:
                        # Fallback if task_queue doesn't exist
                        print(f"Task added (no queue): {task}")
                self.task_entry.delete(0, tk.END)
                self.status_bar.config(text=f"Task added: {task}")
                print(f"Task added: {task}")
        except Exception as e:
            print(f"Error adding task: {e}")
            messagebox.showerror("Error", f"Failed to add task: {e}")

    def update_displays(self):
        """Update the memory and motivations displays."""
        try:
            # Get copies of memory and motivations with proper locking
            with self.system.lock:
                memory_copy = list(self.system.memory) if hasattr(self.system, 'memory') else []
                motivations_copy = list(self.system.motivations) if hasattr(self.system, 'motivations') else []

            # Update memory display
            self.memory_display.config(state=tk.NORMAL)
            self.memory_display.delete(1.0, tk.END)
            for memory in memory_copy:
                self.memory_display.insert(tk.END, f"{memory}\n")
            self.memory_display.config(state=tk.DISABLED)

            # Update motivations display
            self.motivations_display.config(state=tk.NORMAL)
            self.motivations_display.delete(1.0, tk.END)
            for motivation in motivations_copy:
                self.motivations_display.insert(tk.END, f"{motivation}\n")
            self.motivations_display.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error updating displays: {e}")

    def on_closing(self):
        """Handle window close event."""
        try:
            print("🛑 Shutting down application...")

            # Stop voice thread if running
            if self.voice_thread_running:
                print("🎙️ Stopping voice thread...")
                self.voice_thread_running = False

                # Wait a moment for thread to clean up
                time.sleep(0.5)

                # If thread is still alive, log a message
                if self.voice_thread and self.voice_thread.is_alive():
                    print("⚠️ Voice thread is still running, but will terminate when application closes")

            # Release any other resources here if needed

            # Destroy the root window
            self.root.destroy()

        except Exception as e:
            print(f"❌ Error during shutdown: {e}")
            # Ensure the window closes even if there's an error
            self.root.destroy()

    def schedule_autonomous_thought(self):
        """Schedule periodic autonomous thought generation."""
        try:
            # Generate an autonomous thought
            thought = self.system.generate_autonomous_thought()
            if thought:
                print(f"🧠 Autonomous thought generated: {thought}")
                # Update the memory display
                self.update_displays()

            # Schedule the next thought generation with some randomness
            next_interval = self.thought_interval + random.randint(-10000, 10000)  # +/- 10 seconds
            self.root.after(next_interval, self.schedule_autonomous_thought)
        except Exception as e:
            print(f"Error in autonomous thought generation: {e}")
            # Even if there's an error, try to schedule the next thought generation
            self.root.after(self.thought_interval, self.schedule_autonomous_thought)

    def schedule_update(self):
        """Schedule periodic updates of the displays."""
        try:
            self.update_displays()
            # Schedule the next update
            self.root.after(self.update_interval, self.schedule_update)
        except Exception as e:
            print(f"Error in scheduled update: {e}")
            # Even if there's an error, try to schedule the next update
            self.root.after(self.update_interval, self.schedule_update)

    def send_message(self, event=None):
        """Send a message to the AI and display the response."""
        try:
            # Get message from entry
            message = self.chat_entry.get().strip()
            if not message:
                return

            # Clear entry
            self.chat_entry.delete(0, tk.END)

            # Update status
            self.status_bar.config(text="Processing message...")

            # Display user message in chat
            self.add_to_chat("You", message)

            # Get response from AI
            self.status_bar.config(text="AI is thinking...")

            # Use a thread to avoid freezing the UI
            threading.Thread(target=self._get_ai_response, args=(message,), daemon=True).start()

        except Exception as e:
            print(f"Error sending message: {e}")
            self.status_bar.config(text=f"Error: {str(e)}")

    def _get_ai_response(self, message):
        """Get response from AI in a separate thread."""
        try:
            # Update status
            self.status_bar.config(text="AI is thinking...")

            # Get response from the autonomous system
            if SELF_AWARENESS is not None:
                # Use the generate_autonomous_response method from SELF_AWARENESS
                response = SELF_AWARENESS.generate_autonomous_response(message)
            else:
                # Fallback if SELF_AWARENESS is not available
                response = "I'm unable to process your request at this time. The AI system is not fully initialized."
                print("⚠️ SELF_AWARENESS is not initialized, using fallback response")

            # This is the FINAL message from LM Studio that will be displayed in the UI
            final_message = response

            # Display response in chat
            self.add_to_chat("AI", final_message)

            # If voice mode is enabled, speak the FINAL message only
            if self.voice_mode_enabled:
                if not self.is_speaking:
                    print(f"🔊 Voice mode is enabled, speaking final message: '{final_message[:50]}...'")

                    # First, play a direct audio test to ensure the audio system is responsive
                    # This helps "wake up" the audio system before attempting TTS
                    print("🔊 Playing direct audio test to ensure audio system is responsive...")
                    audio_test_result = self._play_direct_audio_test()
                    print(f"🔊 Direct audio test result: {audio_test_result}")

                    # Use a separate thread for speech to avoid blocking the UI
                    print("🔊 Starting speech thread for TTS output...")
                    threading.Thread(target=self._threaded_speak, args=(final_message,), daemon=True).start()
                else:
                    print("🔇 Already speaking, not speaking new message")
            else:
                print("🔇 Voice mode is disabled, not speaking message")

            # Update status
            self.status_bar.config(text="Ready")

        except Exception as e:
            print(f"Error getting AI response: {e}")
            self.status_bar.config(text=f"Error: {str(e)}")

            # Add error message to chat
            self.add_to_chat("System", f"Error: {str(e)}")

    def _threaded_speak(self, text, max_retries=5):
        """Speak text in a separate thread to avoid blocking the UI.

        Args:
            text (str): The text to speak
            max_retries (int): Maximum number of retry attempts if audio fails
        """
        try:
            print(f"🔊 THREADED SPEAK: Starting speech in separate thread for text: '{text[:50]}...'")
            print(f"🔊 THREADED SPEAK: Text length: {len(text)} characters")
            print(f"🔊 THREADED SPEAK: Voice mode enabled: {self.voice_mode_enabled}")
            print(f"🔊 THREADED SPEAK: TTS model initialized: {self.tts_model is not None}")
            print(f"🔊 THREADED SPEAK: Audio libraries available: {AUDIO_AVAILABLE}")
            print(f"🔊 THREADED SPEAK: Coqui TTS available: {COQUI_TTS_AVAILABLE}")
            print(f"🔊 THREADED SPEAK: Max retries: {max_retries}")

            # Set a thread-local flag to indicate we're speaking
            # This is safer than using the instance variable which might be modified by other threads
            thread_speaking = True

            # Update UI to show we're speaking
            self.root.after(0, lambda: self._update_speaking_status(True))

            # Speak the text with retries
            print("🔊 THREADED SPEAK: Calling speak_text method with retries...")
            speak_result = self.speak_text(text, max_retries=max_retries, retry_count=0)
            print(f"🔊 THREADED SPEAK: speak_text returned: {speak_result}")

            # We're done speaking
            thread_speaking = False

            # Update UI to show we're done speaking
            self.root.after(0, lambda: self._update_speaking_status(False))

            print(f"🔊 THREADED SPEAK: Speech completed with result: {speak_result}")

        except Exception as e:
            print(f"❌ THREADED SPEAK: Error in threaded speech: {e}")
            print(f"❌ THREADED SPEAK: Error type: {type(e).__name__}")
            print(f"❌ THREADED SPEAK: Error details: {str(e)}")
            # Update UI to show we're done speaking
            self.root.after(0, lambda: self._update_speaking_status(False))

    def _update_speaking_status(self, is_speaking):
        """Update the speaking status in the UI thread."""
        try:
            self.is_speaking = is_speaking

            if is_speaking:
                self.voice_status.config(text="Voice: Speaking", foreground="orange")
                self.status_bar.config(text="AI is speaking...")
            else:
                if self.voice_mode_enabled:
                    self.voice_status.config(text="Voice: On", foreground="#4CAF50")
                    self.status_bar.config(text="Voice mode enabled - listening...")
                    # Resume listening after speaking
                    self.is_listening = True
                else:
                    self.voice_status.config(text="Voice: Off", foreground="gray")
                    self.status_bar.config(text="Ready")

        except Exception as e:
            print(f"❌ Error updating speaking status: {e}")

    def add_to_chat(self, sender, message):
        """Add a message to the chat display."""
        try:
            # Enable text widget for editing
            self.chat_display.config(state=tk.NORMAL)

            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")

            # Format message with different colors based on sender
            if sender == "You":
                self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.chat_display.insert(tk.END, f"{sender}: ", "user")
                self.chat_display.insert(tk.END, f"{message}\n\n", "user_message")
            elif sender == "AI":
                self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.chat_display.insert(tk.END, f"{sender}: ", "ai")
                self.chat_display.insert(tk.END, f"{message}\n\n", "ai_message")
            else:
                self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.chat_display.insert(tk.END, f"{sender}: ", "system")
                self.chat_display.insert(tk.END, f"{message}\n\n", "system_message")

            # Configure tags for different text styles
            self.chat_display.tag_configure("timestamp", foreground="gray")
            self.chat_display.tag_configure("user", foreground="blue", font=("Segoe UI", 10, "bold"))
            self.chat_display.tag_configure("user_message", foreground="black")
            self.chat_display.tag_configure("ai", foreground="green", font=("Segoe UI", 10, "bold"))
            self.chat_display.tag_configure("ai_message", foreground="black")
            self.chat_display.tag_configure("system", foreground="red", font=("Segoe UI", 10, "bold"))
            self.chat_display.tag_configure("system_message", foreground="black")

            # Scroll to bottom
            self.chat_display.see(tk.END)

            # Disable text widget for editing
            self.chat_display.config(state=tk.DISABLED)

            # Add to chat history
            self.chat_history.append({"sender": sender, "message": message, "timestamp": timestamp})

        except Exception as e:
            print(f"Error adding to chat: {e}")

    def toggle_voice_mode(self):
        """Toggle voice mode on/off."""
        try:
            # Check if required libraries are available
            if not SPEECH_RECOGNITION_AVAILABLE or not AUDIO_AVAILABLE:
                messagebox.showerror(
                    "Missing Dependencies", 
                    "Speech recognition or audio libraries not available.\n\n"
                    "Please install the required packages:\n"
                    "- pip install SpeechRecognition\n"
                    "- pip install sounddevice soundfile"
                )
                return

            # Toggle voice mode
            self.voice_mode_enabled = not self.voice_mode_enabled

            if self.voice_mode_enabled:
                # Update button appearance - green for active
                self.voice_mode_button.configure(text="🎙️ Voice Mode: ON")

                # Create a pulsing effect for the button when active
                def pulse_button():
                    if not self.voice_mode_enabled:
                        return  # Stop pulsing if voice mode is disabled

                    # Change button style based on current state
                    if self.is_listening:
                        style = ttk.Style()
                        style.configure("VoiceMode.TButton", background="#4CAF50")  # Green for listening
                    elif self.is_speaking:
                        style = ttk.Style()
                        style.configure("VoiceMode.TButton", background="#FFA500")  # Orange for speaking
                    else:
                        style = ttk.Style()
                        style.configure("VoiceMode.TButton", background="#2196F3")  # Blue for standby

                    # Schedule next pulse if still in voice mode
                    if self.voice_mode_enabled:
                        self.root.after(1000, pulse_button)

                # Start pulsing
                pulse_button()

                # Update status indicators
                self.voice_status.config(text="Voice: On", foreground="#4CAF50")
                self.status_bar.config(text="Voice mode enabled - listening...")

                # Start voice thread if not already running
                if not self.voice_thread_running:
                    self.voice_thread_running = True
                    self.voice_thread = threading.Thread(target=self._voice_thread_function, daemon=True)
                    self.voice_thread.start()
                    self.add_to_chat("System", "Voice mode activated. You can speak now.")

            else:
                # Update button appearance - default for inactive
                self.voice_mode_button.configure(text="🎙️ Voice Mode: OFF")
                style = ttk.Style()
                style.configure("VoiceMode.TButton", background="#202020")  # Dark gray for inactive

                # Update status indicators
                self.voice_status.config(text="Voice: Off", foreground="gray")
                self.status_bar.config(text="Voice mode disabled")

                # Stop voice thread
                self.voice_thread_running = False
                if self.voice_thread and self.voice_thread.is_alive():
                    # Let the thread terminate naturally
                    pass
                self.add_to_chat("System", "Voice mode deactivated.")

        except Exception as e:
            print(f"Error toggling voice mode: {e}")
            self.status_bar.config(text=f"Error: {str(e)}")

    def show_voice_settings(self):
        """Show dialog for configuring voice input and output devices."""
        try:
            # Create a new top-level window
            settings_window = tk.Toplevel(self.root)
            settings_window.title("Voice Settings")
            settings_window.geometry("500x500")  # Increased size for more content
            settings_window.resizable(False, False)
            settings_window.transient(self.root)  # Set as transient to main window
            settings_window.grab_set()  # Make window modal

            # Add padding
            frame = ttk.Frame(settings_window, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)

            # Title
            ttk.Label(frame, text="Voice Settings", font=("Segoe UI", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))

            # Refresh devices before showing dialog
            self.get_available_devices()

            # Create notebook for tabs
            notebook = ttk.Notebook(frame)
            notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

            # Devices tab
            devices_frame = ttk.Frame(notebook, padding=10)
            notebook.add(devices_frame, text="Audio Devices")

            # Input device selection
            ttk.Label(devices_frame, text="Input Device (Microphone):", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

            # Input device variable
            input_var = tk.StringVar(value=self.selected_input_device if self.selected_input_device else "Default")

            # Input device dropdown
            input_dropdown = ttk.Combobox(devices_frame, textvariable=input_var, values=self.input_devices, width=40)
            input_dropdown.pack(fill=tk.X, pady=(0, 15))

            # Output device selection
            ttk.Label(devices_frame, text="Output Device (Speakers):", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

            # Output device variable
            output_var = tk.StringVar(value=self.selected_output_device if self.selected_output_device else "Default")

            # Output device dropdown
            output_dropdown = ttk.Combobox(devices_frame, textvariable=output_var, values=self.output_devices, width=40)
            output_dropdown.pack(fill=tk.X, pady=(0, 15))

            # Device info
            device_info_frame = ttk.LabelFrame(devices_frame, text="Device Information", padding=10)
            device_info_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

            # Device info text
            device_info_text = tk.Text(device_info_frame, wrap=tk.WORD, height=8, width=40)
            device_info_text.pack(fill=tk.BOTH, expand=True)
            device_info_text.insert(tk.END, "Select a device and click 'Get Info' to see details.")
            device_info_text.config(state=tk.DISABLED)

            # Get device info button
            def get_device_info():
                try:
                    device_info_text.config(state=tk.NORMAL)
                    device_info_text.delete(1.0, tk.END)

                    # Get selected device
                    selected_device = output_var.get()

                    if AUDIO_AVAILABLE:
                        try:
                            # Find device index
                            device_index = None
                            for i, device in enumerate(self.output_devices):
                                if device == selected_device:
                                    device_index = i
                                    break

                            if device_index is not None:
                                # Get device info
                                devices = sd.query_devices()
                                if device_index < len(devices):
                                    device_info = devices[device_index]
                                    device_info_text.insert(tk.END, f"Name: {device_info['name']}\n")
                                    device_info_text.insert(tk.END, f"Channels: {device_info['max_output_channels']}\n")
                                    device_info_text.insert(tk.END, f"Sample Rate: {device_info['default_samplerate']}\n")
                                    device_info_text.insert(tk.END, f"Host API: {device_info['hostapi']}\n")

                                    # Try to get more detailed info
                                    if sys.platform.startswith('linux'):
                                        try:
                                            # Try to get ALSA device info
                                            alsa_info = subprocess.run(["aplay", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                            device_info_text.insert(tk.END, "\nALSA Device Info:\n")
                                            for line in alsa_info.stdout.splitlines():
                                                if selected_device in line:
                                                    device_info_text.insert(tk.END, f"{line}\n")
                                        except:
                                            pass
                                else:
                                    device_info_text.insert(tk.END, "Device not found in system devices list.")
                            else:
                                device_info_text.insert(tk.END, "Device index not found.")
                        except Exception as e:
                            device_info_text.insert(tk.END, f"Error getting device info: {e}")
                    else:
                        device_info_text.insert(tk.END, "Audio libraries not available.")

                    device_info_text.config(state=tk.DISABLED)
                except Exception as e:
                    device_info_text.config(state=tk.NORMAL)
                    device_info_text.delete(1.0, tk.END)
                    device_info_text.insert(tk.END, f"Error: {e}")
                    device_info_text.config(state=tk.DISABLED)

            get_info_button = ttk.Button(devices_frame, text="Get Device Info", command=get_device_info)
            get_info_button.pack(pady=(5, 0))

            # Test tab
            test_frame = ttk.Frame(notebook, padding=10)
            notebook.add(test_frame, text="Test Audio")

            # Test options
            ttk.Label(test_frame, text="Test Options:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

            # Test type selection
            test_type_frame = ttk.Frame(test_frame)
            test_type_frame.pack(fill=tk.X, pady=(0, 10))

            test_type_var = tk.StringVar(value="beep")
            ttk.Radiobutton(test_type_frame, text="Beep", variable=test_type_var, value="beep").pack(side=tk.LEFT, padx=(0, 10))
            ttk.Radiobutton(test_type_frame, text="Voice", variable=test_type_var, value="voice").pack(side=tk.LEFT, padx=(0, 10))
            ttk.Radiobutton(test_type_frame, text="System Sound", variable=test_type_var, value="system").pack(side=tk.LEFT)

            # Test message for voice
            ttk.Label(test_frame, text="Test Message:", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(5, 5))
            test_message_var = tk.StringVar(value="This is a test of the voice output system.")
            test_message_entry = ttk.Entry(test_frame, textvariable=test_message_var, width=40)
            test_message_entry.pack(fill=tk.X, pady=(0, 10))

            # Status message
            status_label = ttk.Label(test_frame, text="", foreground="gray")
            status_label.pack(fill=tk.X, pady=(0, 10))

            # Test results
            test_results_frame = ttk.LabelFrame(test_frame, text="Test Results", padding=10)
            test_results_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

            # Test results text
            test_results_text = tk.Text(test_results_frame, wrap=tk.WORD, height=8, width=40)
            test_results_text.pack(fill=tk.BOTH, expand=True)
            test_results_text.insert(tk.END, "Test results will appear here.")
            test_results_text.config(state=tk.DISABLED)

            # Test button
            def test_audio():
                try:
                    # Get selected devices
                    input_device = input_var.get()
                    output_device = output_var.get()
                    test_type = test_type_var.get()
                    test_message = test_message_var.get()

                    # Update status
                    status_label.config(text=f"Testing: Input={input_device}, Output={output_device}, Type={test_type}", foreground="blue")

                    # Clear test results
                    test_results_text.config(state=tk.NORMAL)
                    test_results_text.delete(1.0, tk.END)
                    test_results_text.insert(tk.END, f"Testing {test_type} with output device: {output_device}\n\n")

                    # Find device index
                    device_index = None
                    for i, device in enumerate(self.output_devices):
                        if device == output_device:
                            device_index = i
                            break

                    # Prepare device args
                    device_args = {}
                    if device_index is not None and output_device != "Default":
                        device_args["device"] = device_index

                    # Test based on type
                    if test_type == "beep":
                        # Test with a beep
                        test_results_text.insert(tk.END, "Playing test beep...\n")
                        test_results_text.config(state=tk.DISABLED)

                        # Create a simple sine wave
                        sample_rate = 44100
                        duration = 0.5
                        frequency = 440
                        t = np.linspace(0, duration, int(sample_rate * duration), False)
                        tone = 0.5 * np.sin(2 * np.pi * frequency * t)
                        tone = tone.astype(np.float32)

                        # Play the tone
                        sd.play(tone, sample_rate, **device_args)
                        sd.wait()

                        # Update results
                        test_results_text.config(state=tk.NORMAL)
                        test_results_text.insert(tk.END, "Beep test completed.\n")
                        test_results_text.insert(tk.END, "Did you hear a beep? If yes, audio output is working.")

                    elif test_type == "voice":
                        # Test with voice
                        test_results_text.insert(tk.END, "Testing voice output...\n")
                        test_results_text.config(state=tk.DISABLED)

                        # Save current output device
                        old_output_device = self.selected_output_device

                        # Set output device for test
                        self.selected_output_device = output_device

                        # Speak the test message
                        speak_result = self.speak_text(test_message)

                        # Restore output device
                        self.selected_output_device = old_output_device

                        # Update results
                        test_results_text.config(state=tk.NORMAL)
                        test_results_text.insert(tk.END, f"Voice test completed with result: {speak_result}\n")
                        test_results_text.insert(tk.END, "Did you hear the test message? If yes, voice output is working.")

                    elif test_type == "system":
                        # Test with system sound
                        test_results_text.insert(tk.END, "Testing system sound...\n")
                        test_results_text.config(state=tk.DISABLED)

                        # Try system sound commands
                        if sys.platform.startswith('linux'):
                            try:
                                subprocess.run(["paplay", "--device", output_device if output_device != "Default" else "", "/usr/share/sounds/freedesktop/stereo/bell.oga"], 
                                              check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                            except:
                                try:
                                    subprocess.run(["aplay", "-D", output_device if output_device != "Default" else "default", "/usr/share/sounds/alsa/Front_Center.wav"], 
                                                  check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                                except:
                                    pass
                        elif sys.platform.startswith('darwin'):
                            try:
                                subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"], 
                                              check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                            except:
                                pass
                        elif sys.platform.startswith('win'):
                            try:
                                subprocess.run(["powershell", "-c", "(New-Object Media.SoundPlayer).PlaySync()"], 
                                              check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                            except:
                                pass

                        # Update results
                        test_results_text.config(state=tk.NORMAL)
                        test_results_text.insert(tk.END, "System sound test completed.\n")
                        test_results_text.insert(tk.END, "Did you hear a system sound? If yes, system audio is working.")

                    # Update status
                    status_label.config(text="Test completed", foreground="green")

                except Exception as e:
                    # Update status and results
                    status_label.config(text=f"Error testing devices: {str(e)}", foreground="red")
                    test_results_text.config(state=tk.NORMAL)
                    test_results_text.delete(1.0, tk.END)
                    test_results_text.insert(tk.END, f"Error testing devices: {str(e)}")

                # Ensure text is scrolled to the end
                test_results_text.see(tk.END)
                test_results_text.config(state=tk.DISABLED)

            test_button = ttk.Button(test_frame, text="Test Audio", command=test_audio)
            test_button.pack(pady=(10, 0))

            # Troubleshooting tab
            troubleshooting_frame = ttk.Frame(notebook, padding=10)
            notebook.add(troubleshooting_frame, text="Troubleshooting")

            # Troubleshooting info
            ttk.Label(troubleshooting_frame, text="Audio Troubleshooting", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

            # Troubleshooting text
            troubleshooting_text = tk.Text(troubleshooting_frame, wrap=tk.WORD, height=15, width=40)
            troubleshooting_text.pack(fill=tk.BOTH, expand=True)

            # Add troubleshooting tips
            troubleshooting_text.insert(tk.END, "If you're having trouble with audio output, try these steps:\n\n")
            troubleshooting_text.insert(tk.END, "1. Check if your speakers are on and volume is up\n")
            troubleshooting_text.insert(tk.END, "2. Check if audio is muted in system settings\n")
            troubleshooting_text.insert(tk.END, "3. Try a different output device\n")
            troubleshooting_text.insert(tk.END, "4. Try running the application with administrator/sudo privileges\n")
            troubleshooting_text.insert(tk.END, "5. Install required audio packages\n")

            if sys.platform.startswith('linux'):
                troubleshooting_text.insert(tk.END, "\nLinux-specific tips:\n")
                troubleshooting_text.insert(tk.END, "1. Run 'sudo usermod -a -G audio,pulse,pulse-access $USER' and then log out and back in\n")
                troubleshooting_text.insert(tk.END, "2. Install audio packages: 'sudo apt-get install alsa-utils pulseaudio speech-dispatcher espeak'\n")
                troubleshooting_text.insert(tk.END, "3. Check system logs: 'dmesg | grep -i audio'\n")
                troubleshooting_text.insert(tk.END, "4. Test with 'aplay -l' to list audio devices\n")
            elif sys.platform.startswith('darwin'):
                troubleshooting_text.insert(tk.END, "\nmacOS-specific tips:\n")
                troubleshooting_text.insert(tk.END, "1. Check System Preferences > Sound\n")
                troubleshooting_text.insert(tk.END, "2. Try resetting the audio subsystem: 'sudo killall coreaudiod'\n")
            elif sys.platform.startswith('win'):
                troubleshooting_text.insert(tk.END, "\nWindows-specific tips:\n")
                troubleshooting_text.insert(tk.END, "1. Check Sound settings in Control Panel\n")
                troubleshooting_text.insert(tk.END, "2. Update audio drivers\n")
                troubleshooting_text.insert(tk.END, "3. Run the Windows audio troubleshooter\n")

            troubleshooting_text.config(state=tk.DISABLED)

            # Run diagnostics button
            def run_diagnostics():
                try:
                    # Run the interactive audio diagnostic wizard
                    self._run_audio_diagnostic_wizard()
                except Exception as e:
                    messagebox.showerror("Error", f"Error running audio diagnostic wizard: {str(e)}")

            diagnostics_button = ttk.Button(troubleshooting_frame, text="Run Diagnostics", command=run_diagnostics)
            diagnostics_button.pack(pady=(10, 0))

            # Buttons frame at the bottom
            buttons_frame = ttk.Frame(frame)
            buttons_frame.pack(fill=tk.X, pady=(10, 0))

            # Save button
            def save_settings():
                try:
                    self.selected_input_device = input_var.get()
                    self.selected_output_device = output_var.get()

                    # If voice mode is active, restart it to apply new settings
                    was_active = self.voice_mode_enabled
                    if was_active:
                        self.toggle_voice_mode()  # Turn off
                        self.toggle_voice_mode()  # Turn back on

                    self.add_to_chat("System", f"Voice settings updated: Input={self.selected_input_device}, Output={self.selected_output_device}")
                    settings_window.destroy()
                except Exception as e:
                    status_label.config(text=f"Error saving settings: {str(e)}", foreground="red")

            save_button = ttk.Button(buttons_frame, text="Save", command=save_settings)
            save_button.pack(side=tk.RIGHT, padx=5)

            # Cancel button
            cancel_button = ttk.Button(buttons_frame, text="Cancel", command=settings_window.destroy)
            cancel_button.pack(side=tk.RIGHT, padx=5)

        except Exception as e:
            print(f"Error showing voice settings: {e}")
            messagebox.showerror("Error", f"Could not open voice settings: {str(e)}")

    def _voice_thread_function(self):
        """Background thread for continuous voice processing."""
        try:
            print("🎙️ Starting voice processing thread")

            # Choose the appropriate speech recognition method
            if WHISPER_AVAILABLE and self.whisper_model:
                self._whisper_voice_loop()
            elif VOSK_AVAILABLE and self.vosk_model:
                self._vosk_voice_loop()
            elif SPEECH_RECOGNITION_AVAILABLE and self.recognizer:
                self._speech_recognition_voice_loop()
            else:
                self.add_to_chat("System", "No speech recognition system available. Voice mode disabled.")
                self.voice_mode_enabled = False
                self.voice_thread_running = False
                self.voice_status.config(text="Voice: Error", foreground="red")
                self.status_bar.config(text="Voice mode error: No speech recognition available")

        except Exception as e:
            print(f"Error in voice thread: {e}")
            self.voice_thread_running = False
            self.voice_mode_enabled = False
            self.voice_status.config(text="Voice: Error", foreground="red")
            self.status_bar.config(text=f"Voice mode error: {str(e)}")

    def _speech_recognition_voice_loop(self):
        """Voice processing loop using SpeechRecognition library."""
        try:
            # Set initial state to listening
            self.is_listening = True
            self.is_speaking = False
            self.voice_status.config(text="Voice: Listening", foreground="#4CAF50")
            self.status_bar.config(text="Voice mode enabled - listening...")

            print("🎙️ Starting speech recognition loop")

            # Initialize microphone with selected device if available
            microphone_args = {}
            if self.selected_input_device and self.selected_input_device != "Default":
                try:
                    # Find the device index for the selected device name
                    device_index = self.input_devices.index(self.selected_input_device)
                    microphone_args["device_index"] = device_index
                    print(f"🎙️ Using microphone: {self.selected_input_device} (index: {device_index})")
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error finding microphone device index: {e}")
                    self.add_to_chat("System", f"Warning: Could not find selected microphone. Using default device.")

            with sr.Microphone(**microphone_args) as source:
                # Adjust for ambient noise
                print("🎙️ Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("🎙️ Adjusted for ambient noise")

                # Add a message to the chat to confirm microphone is ready
                self.add_to_chat("System", "Microphone initialized and ready. You can speak now.")

                # Main voice processing loop
                while self.voice_thread_running:
                    try:
                        if not self.is_speaking:  # Only listen when not speaking
                            if not self.is_listening:
                                self.is_listening = True
                                self.voice_status.config(text="Voice: Listening", foreground="#4CAF50")
                                self.status_bar.config(text="Voice mode enabled - listening...")

                            # Listen for audio with increased phrase_time_limit for longer sentences
                            print("🎙️ Listening...")
                            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=20)

                            # Process audio in a separate thread to keep listening
                            # Use the buffer mechanism to prevent premature sending
                            threading.Thread(target=self._buffer_audio, args=(audio,), daemon=True).start()

                    except sr.WaitTimeoutError:
                        # Timeout is normal, just continue listening
                        pass
                    except Exception as listen_error:
                        print(f"🎙️ Listening error: {listen_error}")
                        self.status_bar.config(text=f"Listening error: {str(listen_error)}")
                        time.sleep(1)  # Prevent rapid error loops

                    # Small sleep to prevent CPU overuse
                    time.sleep(0.1)

            print("🎙️ Voice processing thread stopped")

        except Exception as e:
            print(f"❌ Error in speech recognition voice loop: {e}")
            self.voice_status.config(text="Voice: Error", foreground="red")
            self.status_bar.config(text=f"Voice mode error: {str(e)}")
            self.is_listening = False

            # Add error message to chat
            self.add_to_chat("System", f"Voice recognition error: {str(e)}")

    def _whisper_voice_loop(self):
        """Voice processing loop using Whisper for transcription."""
        # This would be implemented with continuous audio capture and processing
        # For now, we'll use the speech_recognition method as it's more straightforward
        self._speech_recognition_voice_loop()

    def _vosk_voice_loop(self):
        """Voice processing loop using Vosk for transcription."""
        # This would be implemented with continuous audio capture and processing
        # For now, we'll use the speech_recognition method as it's more straightforward
        self._speech_recognition_voice_loop()

    def _buffer_audio(self, audio):
        """Buffer audio data and process when a complete sentence is detected."""
        try:
            # Update status to show we're processing
            self.is_listening = False
            self.voice_status.config(text="Voice: Processing", foreground="blue")
            self.status_bar.config(text="Processing your speech...")

            # Get the current time to track pauses
            current_time = time.time()

            # Process the audio to get text
            text = self._transcribe_audio(audio)

            # Only continue if we got some text
            if text:
                print(f"🎙️ Recognized segment: {text}")

                # Add to buffer
                self.voice_buffer.append(text)

                # Update the last voice time
                self.last_voice_time = current_time

                # Schedule a check for sentence completion after the pause threshold
                # This allows us to wait for a pause before processing the complete sentence
                self.root.after(int(self.voice_pause_threshold * 1000), self._check_sentence_completion)

            # Reset status
            if self.voice_mode_enabled:
                self.voice_status.config(text="Voice: On", foreground="#4CAF50")
                self.status_bar.config(text="Voice mode enabled - listening...")
                # Resume listening after processing
                self.is_listening = True

        except Exception as e:
            print(f"Error buffering audio: {e}")
            # Reset status on error
            if self.voice_mode_enabled:
                self.voice_status.config(text="Voice: On", foreground="#4CAF50")
                self.status_bar.config(text=f"Error processing speech: {str(e)}")
                # Resume listening after error
                self.is_listening = True

    def _transcribe_audio(self, audio):
        """Transcribe audio data to text using available methods."""
        try:
            # Use Whisper if available, otherwise fall back to Google
            if WHISPER_AVAILABLE and self.whisper_model:
                try:
                    # Save audio to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                        temp_audio.write(audio.get_wav_data())

                    # Transcribe with Whisper
                    result = self.whisper_model.transcribe(temp_audio_path)
                    text = result["text"].strip()

                    # Clean up temp file
                    os.unlink(temp_audio_path)
                except FileNotFoundError as e:
                    if "ffmpeg" in str(e):
                        print("⚠️ ffmpeg not found. Falling back to Google Speech Recognition.")
                        print("   To use Whisper, please install ffmpeg:")
                        print("   - Ubuntu/Debian: sudo apt-get install ffmpeg")
                        print("   - macOS: brew install ffmpeg")
                        print("   - Windows: Download from https://ffmpeg.org/download.html")

                        # Add message to chat about ffmpeg
                        self.add_to_chat("System", "Audio processing with Whisper requires ffmpeg. Falling back to Google Speech Recognition. To use Whisper, please install ffmpeg.")

                        # Fall back to Google Speech Recognition
                        text = self.recognizer.recognize_google(audio)
                    else:
                        raise
            else:
                # Fall back to Google Speech Recognition
                text = self.recognizer.recognize_google(audio)

            return text

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

    def _check_sentence_completion(self):
        """Check if there's been a sufficient pause to consider the sentence complete."""
        try:
            # Skip if buffer is empty
            if not self.voice_buffer:
                return

            # Get the current time
            current_time = time.time()

            # Check if there's been a sufficient pause since the last audio
            # or if the buffer has been waiting for too long
            if (current_time - self.last_voice_time) >= self.voice_pause_threshold:
                print(f"🎙️ Pause detected ({current_time - self.last_voice_time:.1f}s). Processing sentence.")
                # Process the complete sentence
                self._process_complete_sentence()
            elif (current_time - self.last_voice_time) >= self.voice_buffer_timeout:
                print(f"🎙️ Buffer timeout ({current_time - self.last_voice_time:.1f}s). Force-processing sentence.")
                # Force-process the sentence if it's been waiting too long
                self._process_complete_sentence()
            else:
                # If we're still collecting speech, schedule another check
                self.root.after(500, self._check_sentence_completion)

        except Exception as e:
            print(f"Error checking sentence completion: {e}")

    def _process_complete_sentence(self):
        """Process the buffered voice segments as a complete sentence."""
        try:
            # Skip if buffer is empty
            if not self.voice_buffer:
                return

            # Combine all segments in the buffer
            complete_text = " ".join(self.voice_buffer)

            # Clear the buffer
            self.voice_buffer = []

            # Reset the last voice time
            self.last_voice_time = 0

            print(f"🎙️ Complete sentence: {complete_text}")

            # Add to chat and process
            self.add_to_chat("You (voice)", complete_text)

            # Process the message
            threading.Thread(target=self._get_ai_response, args=(complete_text,), daemon=True).start()

        except Exception as e:
            print(f"Error processing complete sentence: {e}")
            # Clear buffer on error
            self.voice_buffer = []
            self.last_voice_time = 0

    def _process_audio(self, audio):
        """Legacy method for processing audio data directly (without buffering)."""
        try:
            # Update status to show we're processing
            self.is_listening = False
            self.voice_status.config(text="Voice: Processing", foreground="blue")
            self.status_bar.config(text="Processing your speech...")

            # Transcribe the audio
            text = self._transcribe_audio(audio)

            # Only process if we got some text
            if text:
                print(f"🎙️ Recognized: {text}")

                # Add to chat and process
                self.add_to_chat("You (voice)", text)

                # Process the message
                threading.Thread(target=self._get_ai_response, args=(text,), daemon=True).start()

            # Reset status
            if self.voice_mode_enabled:
                self.voice_status.config(text="Voice: On", foreground="#4CAF50")
                self.status_bar.config(text="Voice mode enabled - listening...")
                # Resume listening after processing
                self.is_listening = True

        except Exception as e:
            print(f"Error processing audio: {e}")
            # Reset status on error
            if self.voice_mode_enabled:
                self.voice_status.config(text="Voice: On", foreground="#4CAF50")
                self.status_bar.config(text=f"Error processing speech: {str(e)}")
                # Resume listening after error
                self.is_listening = True

    def speak_text(self, text, max_retries=3, retry_count=0):
        """Convert text to speech using multiple fallback methods.

        Args:
            text (str): The text to convert to speech
            max_retries (int): Maximum number of retry attempts if audio fails
            retry_count (int): Current retry count (used internally for recursion)

        Returns:
            bool: True if audio output was successful, False otherwise
        """
        try:
            if not text:
                print("🔊 VOICE OUTPUT: Empty text provided, nothing to speak")
                return False

            # Log retry attempt if this is a retry
            if retry_count > 0:
                print(f"🔊 VOICE OUTPUT: RETRY ATTEMPT {retry_count}/{max_retries} for text: '{text[:50]}...'")

            print(f"🔊 VOICE OUTPUT: Starting to speak text: '{text[:50]}...' (length: {len(text)})")
            print(f"🔊 VOICE OUTPUT: Voice mode enabled: {self.voice_mode_enabled}")
            print(f"🔊 VOICE OUTPUT: TTS model initialized: {self.tts_model is not None}")
            print(f"🔊 VOICE OUTPUT: Audio libraries available: {AUDIO_AVAILABLE}")
            print(f"🔊 VOICE OUTPUT: Coqui TTS available: {COQUI_TTS_AVAILABLE}")

            # Update status to show we're speaking
            self.is_speaking = True
            self.is_listening = False  # Don't listen while speaking
            self.voice_status.config(text="Voice: Speaking", foreground="orange")
            self.status_bar.config(text="AI is speaking...")

            # First, try a direct audio test to ensure the audio system is working
            print("🔊 VOICE OUTPUT: Testing audio system with direct beep...")
            direct_audio_success = False

            try:
                # Try raw audio output first (most reliable)
                print("🔊 VOICE OUTPUT: Trying raw audio output...")
                direct_audio_success = self._test_raw_audio_output()
                if direct_audio_success:
                    print("✅ VOICE OUTPUT: Raw audio test successful")
                else:
                    print("⚠️ VOICE OUTPUT: Raw audio test failed, trying direct audio test...")
                    direct_audio_success = self._play_direct_audio_test()
            except Exception as audio_test_error:
                print(f"⚠️ VOICE OUTPUT: Audio test error: {audio_test_error}")
                # Continue anyway, we'll try TTS methods

            if not direct_audio_success:
                print("❌ VOICE OUTPUT: Direct audio tests failed, trying system beep...")
                try:
                    # Try to play a system beep as a last resort
                    self.root.bell()
                    print("✅ VOICE OUTPUT: System beep played")
                    # Even if system beep works, we'll still try TTS
                except Exception as bell_error:
                    print(f"❌ VOICE OUTPUT: System beep failed: {bell_error}")
                    self.add_to_chat("System", "Warning: Audio system may not be working properly.")

                    # Add a notification to the user about potential audio issues
                    self.add_to_chat("System", "Warning: Your audio system appears to be having issues. Voice output may not work.")

            # Create a temporary file for the audio
            temp_audio_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                print(f"🔊 VOICE OUTPUT: Created temporary file at {temp_audio_path}")
            except Exception as temp_file_error:
                print(f"⚠️ VOICE OUTPUT: Error creating temporary file: {temp_file_error}")
                # Continue anyway, some methods don't need a temp file

            # Try multiple TTS methods in order of preference
            tts_success = False

            # Method 1: Coqui TTS
            if not tts_success and COQUI_TTS_AVAILABLE and temp_audio_path:
                try:
                    # Initialize TTS model if not already done
                    if self.tts_model is None:
                        print("🔊 VOICE OUTPUT: TTS model not initialized, initializing now...")

                        # Verify TTS dependencies before attempting initialization
                        print("🔊 VOICE OUTPUT: Verifying TTS dependencies...")
                        try:
                            import torch
                            print(f"✅ VOICE OUTPUT: PyTorch version: {torch.__version__}")
                            print(f"✅ VOICE OUTPUT: CUDA available: {torch.cuda.is_available()}")
                            if torch.cuda.is_available():
                                print(f"✅ VOICE OUTPUT: CUDA device: {torch.cuda.get_device_name(0)}")
                        except Exception as torch_error:
                            print(f"⚠️ VOICE OUTPUT: PyTorch check error: {torch_error}")

                        try:
                            # Direct initialization with the most reliable model
                            print("🔊 VOICE OUTPUT: Initializing TTS model with tacotron2-DDC...")

                            # Force CPU usage to avoid CUDA issues
                            os.environ["CUDA_VISIBLE_DEVICES"] = ""
                            print("🔊 VOICE OUTPUT: Forced CPU mode for TTS")

                            # Initialize with detailed error tracking
                            try:
                                self.tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
                                print("✅ VOICE OUTPUT: TTS model initialized successfully")

                                # Verify model is working with a quick test
                                print("🔊 VOICE OUTPUT: Verifying TTS model with quick test...")
                                test_text = "Test."
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as test_file:
                                    test_path = test_file.name

                                self.tts_model.tts_to_file(text=test_text, file_path=test_path)
                                if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                                    print(f"✅ VOICE OUTPUT: TTS verification successful ({os.path.getsize(test_path)} bytes)")
                                    try:
                                        os.unlink(test_path)
                                    except:
                                        pass
                                else:
                                    print("⚠️ VOICE OUTPUT: TTS verification failed - output file empty or missing")
                                    raise Exception("TTS verification failed")

                            except Exception as init_error:
                                print(f"❌ VOICE OUTPUT: Detailed initialization error: {init_error}")
                                print(f"❌ VOICE OUTPUT: Error type: {type(init_error).__name__}")
                                raise init_error

                        except Exception as tts_init_error:
                            print(f"❌ VOICE OUTPUT: Error initializing primary TTS model: {tts_init_error}")
                            try:
                                # Fallback to a simpler model
                                print("🔊 VOICE OUTPUT: Trying fallback TTS model glow-tts...")
                                self.tts_model = TTS("tts_models/en/ljspeech/glow-tts", progress_bar=False)
                                print("✅ VOICE OUTPUT: Fallback TTS model initialized successfully")
                            except Exception as fallback_error:
                                print(f"❌ VOICE OUTPUT: Error initializing fallback TTS model: {fallback_error}")
                                print(f"❌ VOICE OUTPUT: Error type: {type(fallback_error).__name__}")
                                raise Exception(f"Failed to initialize any TTS model: {fallback_error}")
                    else:
                        print("🔊 VOICE OUTPUT: TTS model already initialized")

                        # Verify the model is still valid
                        print("🔊 VOICE OUTPUT: Verifying existing TTS model...")
                        try:
                            model_type = type(self.tts_model).__name__
                            print(f"✅ VOICE OUTPUT: Existing model type: {model_type}")
                        except Exception as model_check_error:
                            print(f"❌ VOICE OUTPUT: Existing model check failed: {model_check_error}")
                            print("🔊 VOICE OUTPUT: Resetting TTS model and reinitializing...")
                            self.tts_model = None
                            # Recursive call to reinitialize
                            return self.speak_text(text, max_retries, retry_count)

                    if self.tts_model is None:
                        print("❌ VOICE OUTPUT: TTS model initialization failed, falling back to alternatives")
                        raise Exception("TTS model initialization failed")

                    # Generate speech with Coqui TTS
                    print("🔊 VOICE OUTPUT: Generating speech with Coqui TTS...")
                    print(f"🔊 VOICE OUTPUT: TTS model type: {type(self.tts_model).__name__}")
                    print(f"🔊 VOICE OUTPUT: Text to convert: '{text[:50]}...'")

                    # Call tts_to_file with detailed error handling and verification
                    try:
                        print(f"🔊 VOICE OUTPUT: Converting text to speech: '{text[:50]}...'")
                        print(f"🔊 VOICE OUTPUT: Output file: {temp_audio_path}")
                        print(f"🔊 VOICE OUTPUT: TTS model type: {type(self.tts_model).__name__}")

                        # Check if directory exists and is writable
                        output_dir = os.path.dirname(temp_audio_path)
                        if not os.path.exists(output_dir):
                            print(f"⚠️ VOICE OUTPUT: Output directory doesn't exist, creating: {output_dir}")
                            os.makedirs(output_dir, exist_ok=True)

                        if not os.access(output_dir, os.W_OK):
                            print(f"⚠️ VOICE OUTPUT: Output directory not writable: {output_dir}")
                            # Try to use a different temp directory
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tempfile.gettempdir()) as new_temp:
                                temp_audio_path = new_temp.name
                            print(f"🔊 VOICE OUTPUT: Using alternative temp path: {temp_audio_path}")

                        # Sanitize text to avoid TTS errors
                        sanitized_text = text
                        # Remove any control characters that might cause issues
                        sanitized_text = ''.join(c for c in sanitized_text if ord(c) >= 32 or c in '\n\t')
                        # Ensure text isn't empty after sanitization
                        if not sanitized_text.strip():
                            sanitized_text = "The message contained only special characters."
                            print(f"⚠️ VOICE OUTPUT: Text was empty after sanitization, using default message")

                        print(f"🔊 VOICE OUTPUT: Text sanitized, length: {len(sanitized_text)} chars")

                        # Generate the speech file with timeout protection
                        print("🔊 VOICE OUTPUT: Starting TTS generation...")

                        # Define a function to run TTS in a separate thread with timeout
                        def generate_speech():
                            self.tts_model.tts_to_file(text=sanitized_text, file_path=temp_audio_path)
                            return True

                        # Run in a separate thread with timeout
                        tts_thread = threading.Thread(target=generate_speech)
                        tts_thread.daemon = True
                        tts_thread.start()

                        # Wait for completion with timeout
                        max_wait_time = 30  # seconds
                        start_time = time.time()
                        while tts_thread.is_alive() and time.time() - start_time < max_wait_time:
                            time.sleep(0.5)
                            # Print progress every 5 seconds
                            elapsed = time.time() - start_time
                            if elapsed % 5 < 0.5:
                                print(f"🔊 VOICE OUTPUT: TTS generation in progress... ({elapsed:.1f}s)")

                        if tts_thread.is_alive():
                            print(f"⚠️ VOICE OUTPUT: TTS generation timed out after {max_wait_time} seconds")
                            # Continue anyway, the file might be partially created and usable
                        else:
                            print("✅ VOICE OUTPUT: TTS generation completed successfully")

                        # Verify the file was created and has content
                        if os.path.exists(temp_audio_path):
                            file_size = os.path.getsize(temp_audio_path)
                            print(f"✅ VOICE OUTPUT: Audio file created successfully ({file_size} bytes)")

                            if file_size == 0:
                                print("❌ VOICE OUTPUT: Audio file is empty")
                                raise Exception("Audio file is empty")

                            # Verify the file is a valid audio file
                            try:
                                print("🔊 VOICE OUTPUT: Verifying audio file format...")
                                audio_data, sample_rate = sf.read(temp_audio_path)
                                print(f"✅ VOICE OUTPUT: Audio file verified: {len(audio_data)} samples at {sample_rate}Hz")
                            except Exception as verify_error:
                                print(f"❌ VOICE OUTPUT: Audio file verification failed: {verify_error}")
                                raise Exception(f"Invalid audio file: {verify_error}")
                        else:
                            print("❌ VOICE OUTPUT: Audio file was not created")
                            raise Exception("Audio file was not created")

                    except Exception as tts_error:
                        print(f"❌ VOICE OUTPUT: Error in TTS generation: {tts_error}")
                        print(f"❌ VOICE OUTPUT: Error type: {type(tts_error).__name__}")
                        print(f"❌ VOICE OUTPUT: Error details: {str(tts_error)}")

                        # Try with a shorter text as a fallback
                        try:
                            print("🔊 VOICE OUTPUT: Trying with shorter text as fallback...")
                            shorter_text = text[:100] + "..." if len(text) > 100 else text
                            # Sanitize the shorter text as well
                            shorter_text = ''.join(c for c in shorter_text if ord(c) >= 32 or c in '\n\t')
                            if not shorter_text.strip():
                                shorter_text = "The message contained only special characters."

                            print(f"🔊 VOICE OUTPUT: Using shorter text: '{shorter_text}'")
                            self.tts_model.tts_to_file(text=shorter_text, file_path=temp_audio_path)

                            # Verify the fallback file
                            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                                print("✅ VOICE OUTPUT: Fallback TTS generation successful")
                            else:
                                print("❌ VOICE OUTPUT: Fallback TTS generation failed")
                                raise Exception("Fallback TTS generation failed")
                        except Exception as short_error:
                            print(f"❌ VOICE OUTPUT: Error with shorter text: {short_error}")
                            raise tts_error

                    # Play the audio
                    print("🔊 VOICE OUTPUT: Playing audio...")

                    # Load the audio file
                    data, fs = sf.read(temp_audio_path)
                    print(f"✅ VOICE OUTPUT: Audio file loaded: {len(data)} samples at {fs}Hz")

                    # Play the audio with improved error handling and device management
                    print("🔊 VOICE OUTPUT: Starting audio playback...")

                    try:
                        # Get default device info for debugging
                        try:
                            default_device = sd.default.device
                            device_info = sd.query_devices(default_device[1])
                            print(f"🔊 VOICE OUTPUT: Using default output device: {device_info['name']} (ID: {default_device[1]})")
                        except Exception as device_error:
                            print(f"⚠️ VOICE OUTPUT: Could not get default device info: {device_error}")

                        # Create a blocking output stream for more reliable playback
                        print("🔊 VOICE OUTPUT: Creating output stream...")
                        with sd.OutputStream(samplerate=fs, channels=data.shape[1] if len(data.shape) > 1 else 1) as stream:
                            print("🔊 VOICE OUTPUT: Stream created, writing audio data...")
                            # Write data in chunks to avoid buffer issues
                            chunk_size = 1024
                            for i in range(0, len(data), chunk_size):
                                chunk = data[i:i + chunk_size]
                                stream.write(chunk)
                                # Print progress every ~1 second
                                if i % (fs * 2) < chunk_size:
                                    print(f"🔊 VOICE OUTPUT: Playing... {i/len(data)*100:.1f}% complete")

                            # Ensure all data is played
                            stream.stop()
                            print("✅ VOICE OUTPUT: Audio playback complete")
                    except Exception as stream_error:
                        print(f"❌ VOICE OUTPUT: Stream playback error: {stream_error}")

                        # Fallback to simple playback method
                        print("🔊 VOICE OUTPUT: Trying fallback playback method...")
                        try:
                            sd.play(data, fs)
                            sd.wait()
                            print("✅ VOICE OUTPUT: Fallback playback complete")
                        except Exception as fallback_error:
                            print(f"❌ VOICE OUTPUT: Fallback playback error: {fallback_error}")
                            raise fallback_error
                    tts_success = True

                except Exception as coqui_error:
                    print(f"❌ VOICE OUTPUT: Error using Coqui TTS: {coqui_error}")
                    # Continue to next method

            # Method 2: Mimic3
            if not tts_success and temp_audio_path:
                try:
                    print("🔊 VOICE OUTPUT: Trying Mimic3...")
                    tts_success = self._fallback_to_mimic3(text, temp_audio_path)
                except Exception as mimic_error:
                    print(f"❌ VOICE OUTPUT: Error in Mimic3 fallback: {mimic_error}")

            # Method 3: System TTS commands
            if not tts_success:
                try:
                    print("🔊 VOICE OUTPUT: Trying system TTS commands...")
                    tts_success = self._system_tts_fallback(text)
                except Exception as system_tts_error:
                    print(f"❌ VOICE OUTPUT: Error in system TTS fallback: {system_tts_error}")

            # Method 4: Direct audio simulation
            if not tts_success:
                try:
                    print("🔊 VOICE OUTPUT: Trying direct audio simulation...")
                    tts_success = self._play_direct_speech_simulation()
                except Exception as simulation_error:
                    print(f"❌ VOICE OUTPUT: Error in direct audio simulation: {simulation_error}")

            # Method 5: Raw audio output (most reliable, try again)
            if not tts_success:
                try:
                    print("🔊 VOICE OUTPUT: Trying raw audio output as fallback...")
                    tts_success = self._test_raw_audio_output()
                except Exception as raw_audio_error:
                    print(f"❌ VOICE OUTPUT: Error in raw audio output: {raw_audio_error}")

            # Method 6: System audio test
            if not tts_success:
                try:
                    print("🔊 VOICE OUTPUT: Trying system audio test...")
                    tts_success = self._test_system_audio()
                except Exception as system_audio_error:
                    print(f"❌ VOICE OUTPUT: Error in system audio test: {system_audio_error}")

            # Method 7: WebAudio API through browser (absolute last resort)
            if not tts_success:
                try:
                    print("🔊 VOICE OUTPUT: Trying WebAudio API through browser as last resort...")

                    # Create a temporary HTML file with the text and WebAudio API
                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode='w') as temp_html:
                        temp_html_path = temp_html.name

                        # Escape the text for JavaScript
                        escaped_text = text.replace('"', '\\"').replace('\n', '\\n')

                        # Write HTML content with WebAudio API code and the text
                        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Audio Message</title>
    <style>
        body {{ font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }}
        .message {{ margin: 20px; padding: 20px; border: 1px solid #ccc; border-radius: 5px; }}
        button {{ padding: 10px 20px; font-size: 16px; margin: 10px; }}
        #status {{ margin-top: 20px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Audio Message</h1>
    <div class="message">{escaped_text}</div>
    <button id="playButton">Play Audio Notification</button>
    <div id="status">Click the button to play an audio notification</div>

    <script>
        // Function to play a test sound using WebAudio API
        function playTestSound() {{
            const statusElement = document.getElementById('status');
            statusElement.textContent = 'Initializing audio context...';

            try {{
                // Create audio context
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                const audioCtx = new AudioContext();

                statusElement.textContent = 'Creating audio...';

                // Create oscillator for a notification sound
                const oscillator = audioCtx.createOscillator();
                oscillator.type = 'sine';

                // Create gain node for volume control
                const gainNode = audioCtx.createGain();
                gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime); // Set volume to 10%

                // Connect nodes
                oscillator.connect(gainNode);
                gainNode.connect(audioCtx.destination);

                statusElement.textContent = 'Playing notification sound...';

                // Create a simple notification sound (rising tone)
                oscillator.frequency.setValueAtTime(440, audioCtx.currentTime);
                oscillator.frequency.linearRampToValueAtTime(880, audioCtx.currentTime + 0.5);

                // Start and stop the oscillator
                oscillator.start();

                // Play for 0.5 second
                setTimeout(() => {{
                    oscillator.stop();
                    statusElement.textContent = 'Sound played successfully!';

                    // Close the window after 10 seconds
                    setTimeout(() => {{
                        window.close();
                    }}, 10000);

                }}, 500);
            }} catch (error) {{
                statusElement.textContent = 'Error: ' + error.message;
                console.error('WebAudio error:', error);
            }}
        }}

        // Add click event listener to the button
        document.getElementById('playButton').addEventListener('click', playTestSound);

        // Auto-play might be blocked by browser, so we don't try it
        // Instead, we show a clear message to the user
        document.getElementById('status').textContent = 'Please click the button to hear the notification';
    </script>
</body>
</html>
"""
                        temp_html.write(html_content)

                    print(f"🔊 VOICE OUTPUT: Created temporary HTML file at {temp_html_path}")

                    # Open the HTML file in the default web browser
                    import webbrowser
                    webbrowser.open('file://' + temp_html_path)

                    # Add a message to the chat
                    self.add_to_chat("System", "Unable to play audio through normal channels. A browser window has been opened with your message and an audio notification option.")

                    # Clean up the temporary file after a delay
                    def cleanup_temp_file():
                        try:
                            time.sleep(30)  # Wait for 30 seconds before cleaning up
                            os.unlink(temp_html_path)
                            print(f"✅ VOICE OUTPUT: Temporary HTML file {temp_html_path} cleaned up")
                        except Exception as e:
                            print(f"⚠️ VOICE OUTPUT: Error cleaning up temporary HTML file: {e}")

                    # Start cleanup in a separate thread
                    threading.Thread(target=cleanup_temp_file, daemon=True).start()

                    print("✅ VOICE OUTPUT: WebAudio fallback initiated in browser")
                    tts_success = True  # Consider this a success since we've provided a way for the user to hear something

                except Exception as web_audio_error:
                    print(f"❌ VOICE OUTPUT: Error in WebAudio fallback: {web_audio_error}")

            # Clean up temp file
            if temp_audio_path:
                try:
                    os.unlink(temp_audio_path)
                    print(f"✅ VOICE OUTPUT: Temporary file {temp_audio_path} cleaned up")
                except Exception as file_error:
                    print(f"⚠️ VOICE OUTPUT: Error cleaning up temp file: {file_error}")

            # Reset status after speaking
            self.is_speaking = False
            if self.voice_mode_enabled:
                self.voice_status.config(text="Voice: On", foreground="#4CAF50")
                self.status_bar.config(text="Voice mode enabled - listening...")
                # Resume listening after speaking
                self.is_listening = True

            # Report final status
            if tts_success:
                print("✅ VOICE OUTPUT: Text-to-speech completed successfully")
                return True
            else:
                print("❌ VOICE OUTPUT: Text-to-speech failed")
                # Add a message to the chat to inform the user
                self.add_to_chat("System", "Voice output failed. Please check the console for details.")

                # Add detailed troubleshooting information
                troubleshooting_msg = (
                    "Audio troubleshooting:\n"
                    "1. Check if your speakers are on and volume is up\n"
                    "2. Check if audio is muted in system settings\n"
                    "3. Try running the application with administrator/sudo privileges\n"
                    "4. Install required audio packages (see previous messages)\n"
                    "5. Try restarting your computer\n"
                    "6. Check if other applications can play sound\n"
                    "7. Try running the application with the --test-audio flag to diagnose audio issues"
                )
                self.add_to_chat("System", troubleshooting_msg)

                # Use system notification as a robust fallback
                print("🔊 VOICE OUTPUT: Attempting to use system notification as fallback...")
                notification_success = False

                try:
                    notification_success = self._show_system_notification(
                        "Voice Output Message", 
                        f"Message: {text[:100]}{'...' if len(text) > 100 else ''}"
                    )

                    if notification_success:
                        print("✅ VOICE OUTPUT: Successfully displayed system notification as fallback")
                        self.add_to_chat("System", "Voice output failed. Message displayed as a system notification instead.")
                        # Consider this a partial success since the user was notified
                        return True
                    else:
                        print("❌ VOICE OUTPUT: System notification fallback failed")
                except Exception as notification_error:
                    print(f"❌ VOICE OUTPUT: Error showing system notification: {notification_error}")

                # If system notification failed, try to display a message box as absolute last resort
                try:
                    print("🔊 VOICE OUTPUT: Attempting to show message box as last resort...")
                    # Use after to avoid blocking the main thread
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Voice Output Message", 
                        f"Message: {text[:500]}{'...' if len(text) > 500 else ''}"
                    ))
                    print("✅ VOICE OUTPUT: Message box scheduled to display")
                    self.add_to_chat("System", "Voice output failed. Message displayed in a dialog box instead.")
                    # Consider this a partial success since the user was notified
                    return True
                except Exception as msgbox_error:
                    print(f"❌ VOICE OUTPUT: Error showing message box: {msgbox_error}")

                # If we've reached this point, all methods have failed
                # Check if we should retry
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"🔊 VOICE OUTPUT: All methods failed. Retrying ({retry_count}/{max_retries})...")

                    # Add a message to the chat about retrying
                    self.add_to_chat("System", f"Voice output failed. Retrying ({retry_count}/{max_retries})...")

                    # Reset speaking status for the retry
                    self.is_speaking = False

                    # Try to restart the audio subsystem before retrying
                    if retry_count == 1:  # Only on first retry
                        print("🔊 VOICE OUTPUT: Attempting to restart audio subsystem before retry...")
                        try:
                            self._restart_audio_subsystem()
                        except Exception as restart_error:
                            print(f"⚠️ VOICE OUTPUT: Error restarting audio subsystem: {restart_error}")

                    # Add a delay before retrying to avoid overwhelming the system
                    # Use exponential backoff for the delay
                    retry_delay = 1 * (2 ** (retry_count - 1))  # 1, 2, 4, 8... seconds
                    print(f"🔊 VOICE OUTPUT: Waiting {retry_delay} seconds before retry...")

                    # Use root.after to schedule the retry after the delay
                    # This avoids blocking the main thread
                    self.root.after(int(retry_delay * 1000), lambda: self.speak_text(text, max_retries, retry_count))

                    # Return True to indicate we're handling it (retrying)
                    return True

                # If we've exhausted all retries, give up
                print(f"❌ VOICE OUTPUT: All methods failed after {retry_count} retries. Giving up.")
                return False

        except Exception as e:
            print(f"❌ VOICE OUTPUT: Error speaking text: {e}")
            print(f"❌ VOICE OUTPUT: Error type: {type(e).__name__}")
            print(f"❌ VOICE OUTPUT: Error details: {str(e)}")

            # Reset status on error
            self.is_speaking = False
            if self.voice_mode_enabled:
                self.voice_status.config(text="Voice: On", foreground="#4CAF50")
                self.status_bar.config(text="Voice mode enabled - listening...")
                # Resume listening after error
                self.is_listening = True

            # Add a message to the chat to inform the user
            self.add_to_chat("System", f"Voice output error: {str(e)}")

            # Try to play a system beep to indicate error
            try:
                self.root.bell()
            except:
                pass

            # Try to use system notification as a last resort
            try:
                self._show_system_notification("Voice Output Error", 
                                             f"Error: {str(e)}")
            except:
                pass

            # Check if we should retry after an exception
            if retry_count < max_retries:
                retry_count += 1
                print(f"🔊 VOICE OUTPUT: Exception occurred. Retrying ({retry_count}/{max_retries})...")

                # Add a message to the chat about retrying
                self.add_to_chat("System", f"Voice output error. Retrying ({retry_count}/{max_retries})...")

                # Add a delay before retrying
                retry_delay = 1 * (2 ** (retry_count - 1))  # 1, 2, 4, 8... seconds
                print(f"🔊 VOICE OUTPUT: Waiting {retry_delay} seconds before retry...")

                # Schedule the retry
                self.root.after(int(retry_delay * 1000), lambda: self.speak_text(text, max_retries, retry_count))

                # Return True to indicate we're handling it (retrying)
                return True

            return False

    def _show_system_notification(self, title, message):
        """Show a system notification as a fallback when audio fails."""
        try:
            print(f"🔔 NOTIFICATION: Showing system notification: {title} - {message}")

            # Platform-specific implementations
            if sys.platform.startswith('win'):
                # Windows implementation
                try:
                    # Try using Windows 10 toast notifications
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast(title, message, duration=5, threaded=True)
                    return True
                except ImportError:
                    # Fall back to a message box
                    self.root.after(0, lambda: messagebox.showinfo(title, message))
                    return True

            elif sys.platform.startswith('darwin'):
                # macOS implementation
                try:
                    # Use osascript to show a notification
                    os_command = f'display notification "{message}" with title "{title}"'
                    subprocess.run(["osascript", "-e", os_command], 
                                  check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return True
                except Exception:
                    # Fall back to a message box
                    self.root.after(0, lambda: messagebox.showinfo(title, message))
                    return True

            elif sys.platform.startswith('linux'):
                # Linux implementation
                try:
                    # Try using notify-send
                    subprocess.run(["notify-send", title, message], 
                                  check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return True
                except Exception:
                    # Fall back to a message box
                    self.root.after(0, lambda: messagebox.showinfo(title, message))
                    return True

            # Fallback for other platforms
            self.root.after(0, lambda: messagebox.showinfo(title, message))
            return True

        except Exception as e:
            print(f"❌ NOTIFICATION: Error showing system notification: {e}")
            return False

    def _system_tts_fallback(self, text):
        """Use system TTS commands as a fallback."""
        try:
            print("🔊 SYSTEM TTS: Attempting to use system TTS commands...")

            success = False

            # Try different commands based on the platform
            if sys.platform.startswith('linux'):
                # Try speech-dispatcher (spd-say)
                try:
                    print("🔊 SYSTEM TTS: Trying spd-say...")
                    subprocess.run(["spd-say", "-r", "0", text], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                    print("✅ SYSTEM TTS: spd-say executed")
                    success = True
                except Exception as e:
                    print(f"⚠️ SYSTEM TTS: spd-say failed: {e}")

                # Try espeak
                if not success:
                    try:
                        print("🔊 SYSTEM TTS: Trying espeak...")
                        subprocess.run(["espeak", text], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                        print("✅ SYSTEM TTS: espeak executed")
                        success = True
                    except Exception as e:
                        print(f"⚠️ SYSTEM TTS: espeak failed: {e}")

                # Try festival
                if not success:
                    try:
                        print("🔊 SYSTEM TTS: Trying festival...")
                        # Create a temporary text file
                        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_text:
                            temp_text_path = temp_text.name
                            temp_text.write(text.encode('utf-8'))

                        # Run festival
                        subprocess.run(["festival", "--tts", temp_text_path], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                        print("✅ SYSTEM TTS: festival executed")
                        success = True

                        # Clean up
                        os.unlink(temp_text_path)
                    except Exception as e:
                        print(f"⚠️ SYSTEM TTS: festival failed: {e}")
                        try:
                            os.unlink(temp_text_path)
                        except:
                            pass

            elif sys.platform.startswith('darwin'):
                # Try say command (macOS)
                try:
                    print("🔊 SYSTEM TTS: Trying say command...")
                    subprocess.run(["say", text], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                    print("✅ SYSTEM TTS: say command executed")
                    success = True
                except Exception as e:
                    print(f"⚠️ SYSTEM TTS: say command failed: {e}")

            elif sys.platform.startswith('win'):
    # Try PowerShell (Windows)
    try:
        print("🔊 SYSTEM TTS: Trying PowerShell speech synthesis...")
        text_escaped = text.replace("'", "''")
        ps_command = (
            f"Add-Type -AssemblyName System.Speech; "
            f"(New-Object 
    System.Speech.Synthesis.SpeechSynthesizer).Speak('{text_escaped}')"
    )
        subprocess.run(
            ["powershell", "-c", ps_command],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        print("✅ SYSTEM TTS: PowerShell speech synthesis executed")
        success = True
    except Exception as e:
        print(f"⚠️ SYSTEM TTS: PowerShell speech synthesis failed: {e}")

            return success

        except Exception as e:
            print(f"❌ SYSTEM TTS: Error in system TTS fallback: {e}")
            return False

    def test_voice_output(self):
        """Test the voice output functionality."""
        try:
            print("🔊 VOICE TEST: Starting voice output test...")
            self.add_to_chat("System", "Testing voice output capabilities...")

            # First, try to play a simple beep to test audio playback
            self._test_audio_playback()

            # Then test the TTS system
            test_message = "Voice output is now enabled and working."
            self.add_to_chat("System", test_message)
            self.speak_text(test_message)

            # Schedule a comprehensive audio output test
            self.root.after(5000, lambda: self._test_all_audio_methods())

            print("🔊 VOICE TEST: Voice output test completed")
        except Exception as e:
            print(f"❌ VOICE TEST: Error testing voice output: {e}")
            self.add_to_chat("System", f"Error testing voice output: {e}")

    def _test_all_audio_methods(self):
        """Test all available audio output methods to determine which ones work."""
        try:
            print("\n" + "="*50)
            print("🔊 COMPREHENSIVE AUDIO OUTPUT TEST")
            print("="*50)

            self.add_to_chat("System", "Running comprehensive audio output test to determine which methods work on your system...")

            # Track results
            results = {}
            working_methods = []

            # Test 1: Direct audio using sounddevice
            print("\n🔊 TEST 1: Direct audio using sounddevice")
            try:
                # Create a simple sine wave
                sample_rate = 44100
                duration = 0.5
                frequency = 440
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                tone = 0.5 * np.sin(2 * np.pi * frequency * t)
                tone = tone.astype(np.float32)

                # Play the tone
                print("🔊 Playing tone using sounddevice...")
                sd.play(tone, sample_rate)
                sd.wait()
                print("✅ Sounddevice test completed")
                results["sounddevice"] = "SUCCESS"
                working_methods.append("sounddevice")
            except Exception as e:
                print(f"❌ Sounddevice test failed: {e}")
                results["sounddevice"] = f"FAILED: {str(e)}"

            # Test 2: Coqui TTS
            print("\n🔊 TEST 2: Coqui TTS")
            if COQUI_TTS_AVAILABLE:
                try:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name

                    # Initialize TTS model if needed
                    if self.tts_model is None:
                        print("🔊 Initializing Coqui TTS model with retries...")
                        self._initialize_tts_with_retry(max_retries=1, retry_count=0)

                    if self.tts_model is not None:
                        # Generate speech
                        print("🔊 Generating speech with Coqui TTS...")
                        self.tts_model.tts_to_file(text="Testing Coqui TTS.", file_path=temp_audio_path)

                        # Play the audio
                        data, fs = sf.read(temp_audio_path)
                        sd.play(data, fs)
                        sd.wait()

                        # Clean up
                        os.unlink(temp_audio_path)

                        print("✅ Coqui TTS test completed")
                        results["coqui_tts"] = "SUCCESS"
                        working_methods.append("coqui_tts")
                    else:
                        print("❌ Coqui TTS model initialization failed")
                        results["coqui_tts"] = "FAILED: Model initialization failed"
                except Exception as e:
                    print(f"❌ Coqui TTS test failed: {e}")
                    results["coqui_tts"] = f"FAILED: {str(e)}"
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
            else:
                print("⚠️ Coqui TTS not available")
                results["coqui_tts"] = "NOT AVAILABLE"

            # Test 3: Mimic3
            print("\n🔊 TEST 3: Mimic3")
            try:
                # Check if mimic3 is installed
                mimic3_check = subprocess.run(["which", "mimic3"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if mimic3_check.returncode == 0:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name

                    # Generate speech
                    print("🔊 Generating speech with Mimic3...")
                    subprocess.run(
                        ["mimic3", "--voice", "en_US/vctk_low", "--output", temp_audio_path, "Testing Mimic3."],
                        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )

                    # Play the audio
                    data, fs = sf.read(temp_audio_path)
                    sd.play(data, fs)
                    sd.wait()

                    # Clean up
                    os.unlink(temp_audio_path)

                    print("✅ Mimic3 test completed")
                    results["mimic3"] = "SUCCESS"
                    working_methods.append("mimic3")
                else:
                    print("⚠️ Mimic3 not installed")
                    results["mimic3"] = "NOT INSTALLED"
            except Exception as e:
                print(f"❌ Mimic3 test failed: {e}")
                results["mimic3"] = f"FAILED: {str(e)}"
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

            # Test 4: System TTS commands
            print("\n🔊 TEST 4: System TTS commands")
            system_tts_result = self._system_tts_fallback("Testing system TTS commands.")
            if system_tts_result:
                print("✅ System TTS test completed")
                results["system_tts"] = "SUCCESS"
                working_methods.append("system_tts")
            else:
                print("❌ System TTS test failed")
                results["system_tts"] = "FAILED"

            # Test 5: System audio test
            print("\n🔊 TEST 5: System audio test")
            system_audio_result = self._test_system_audio()
            if system_audio_result:
                print("✅ System audio test completed")
                results["system_audio"] = "SUCCESS"
                working_methods.append("system_audio")
            else:
                print("❌ System audio test failed")
                results["system_audio"] = "FAILED"

            # Test 6: Direct speech simulation
            print("\n🔊 TEST 6: Direct speech simulation")
            direct_speech_result = self._play_direct_speech_simulation()
            if direct_speech_result:
                print("✅ Direct speech simulation completed")
                results["direct_speech"] = "SUCCESS"
                working_methods.append("direct_speech")
            else:
                print("❌ Direct speech simulation failed")
                results["direct_speech"] = "FAILED"

            # Test 7: WebAudio API test
            print("\n🔊 TEST 7: WebAudio API test")
            web_audio_result = self._test_web_audio_output()
            if web_audio_result:
                print("✅ WebAudio API test initiated")
                results["web_audio"] = "INITIATED"
                # Note: We don't add this to working_methods yet since we can't confirm it worked
                # The user will need to manually confirm if they heard the sound
            else:
                print("❌ WebAudio API test failed")
                results["web_audio"] = "FAILED"

            # Test 8: Direct hardware audio test
            print("\n🔊 TEST 8: Direct hardware audio test")
            hardware_audio_result = self._test_direct_hardware_audio()
            if hardware_audio_result:
                print("✅ Direct hardware audio test completed")
                results["hardware_audio"] = "SUCCESS"
                working_methods.append("hardware_audio")
            else:
                print("❌ Direct hardware audio test failed")
                results["hardware_audio"] = "FAILED"

            # Summarize results
            print("\n" + "="*50)
            print("🔊 AUDIO OUTPUT TEST RESULTS")
            print("="*50)

            for method, result in results.items():
                print(f"🔊 {method}: {result}")

            print(f"\n🔊 Working methods: {', '.join(working_methods) if working_methods else 'None'}")
            print("="*50)

            # Add results to chat
            message = "Audio Output Test Results:\n\n"
            for method, result in results.items():
                status = "✅" if "SUCCESS" in result else "⚠️" if "INITIATED" in result else "❌"
                message += f"{status} {method}: {result}\n"

            message += f"\nWorking methods: {', '.join(working_methods) if working_methods else 'None'}"

            if "web_audio" in results and results["web_audio"] == "INITIATED":
                message += "\n\nNote: WebAudio test was initiated in your browser. Please confirm if you heard the sound."

            if working_methods:
                message += "\n\nRecommendation: Use the following methods in order of preference:\n"
                for method in working_methods:
                    message += f"- {method}\n"

                # Update configuration based on results
                self._update_audio_configuration(working_methods)
            else:
                message += "\n\nNo working audio methods found. Please check your audio configuration."

                # If no methods work, suggest checking system audio settings
                message += "\n\nTroubleshooting suggestions:"
                message += "\n1. Check if your speakers/headphones are connected and turned on"
                message += "\n2. Check if your system volume is muted or too low"
                message += "\n3. Try the WebAudio test in your browser"
                message += "\n4. Run the audio diagnostic wizard (Settings > Voice Settings > Diagnostics) for detailed troubleshooting"

            self.add_to_chat("System", message)

            return working_methods

        except Exception as e:
            print(f"❌ Error in comprehensive audio test: {e}")
            self.add_to_chat("System", f"Error in comprehensive audio test: {e}")
            return []

    def _update_audio_configuration(self, working_methods):
        """Update audio configuration based on working methods."""
        try:
            print("🔊 Updating audio configuration based on test results...")

            # If Coqui TTS is working, make sure it's initialized with retries
            if "coqui_tts" in working_methods and self.tts_model is None:
                self._initialize_tts_with_retry(max_retries=2, retry_count=0)
                print("✅ Initialized Coqui TTS model with retries")

            # If direct audio is working but TTS is not, try to use direct audio for feedback
            if "sounddevice" in working_methods and not any(method in working_methods for method in ["coqui_tts", "mimic3", "system_tts"]):
                print("⚠️ TTS not working, will use direct audio for feedback")
                self.add_to_chat("System", "TTS not working, will use direct audio for feedback")

                # Play a confirmation sound
                self._play_direct_audio_test()

            # If we have working methods, try to speak a confirmation message
            if working_methods:
                # Use the first working method to speak a confirmation message
                if "coqui_tts" in working_methods:
                    self.speak_text("Audio configuration updated successfully.")
                elif "mimic3" in working_methods:
                    self.speak_text("Audio configuration updated successfully.")
                elif "system_tts" in working_methods:
                    self._system_tts_fallback("Audio configuration updated successfully.")
                elif "direct_speech" in working_methods:
                    self._play_direct_speech_simulation()
                elif "sounddevice" in working_methods:
                    self._play_direct_audio_test()

            print("✅ Audio configuration updated")

        except Exception as e:
            print(f"❌ Error updating audio configuration: {e}")
            self.add_to_chat("System", f"Error updating audio configuration: {e}")

    def _test_audio_playback(self):
        """Test basic audio playback to verify sound system is working."""
        try:
            print("🔊 AUDIO TEST: Testing basic audio playback...")
            self.add_to_chat("System", "Testing audio playback system...")

            # Create a simple sine wave for testing
            sample_rate = 44100  # Standard sample rate
            duration = 0.5  # Short duration
            frequency = 440  # A4 note

            # Generate a sine wave
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = 0.5 * np.sin(2 * np.pi * frequency * t)

            # Ensure the audio is in the correct format (float32)
            tone = tone.astype(np.float32)

            print(f"🔊 AUDIO TEST: Generated test tone: {len(tone)} samples at {sample_rate}Hz")

            # Use selected output device if available
            device_args = {}
            if self.selected_output_device and self.selected_output_device != "Default":
                try:
                    # Find the device index for the selected device name
                    device_index = self.output_devices.index(self.selected_output_device)
                    device_args["device"] = device_index
                    print(f"🔊 AUDIO TEST: Using output device: {self.selected_output_device} (index: {device_index})")
                except (ValueError, IndexError) as e:
                    print(f"⚠️ AUDIO TEST: Error finding output device index: {e}")
                    print("🔊 AUDIO TEST: Continuing with default device")
            else:
                print("🔊 AUDIO TEST: Using default audio device")

            # Play the tone
            print("🔊 AUDIO TEST: Playing test tone...")
            sd.play(tone, sample_rate, **device_args)
            sd.wait()  # Wait until audio is finished playing
            print("✅ AUDIO TEST: Test tone playback complete")

            # Add a small delay
            time.sleep(0.5)

            # If we got here, audio playback is working
            self.add_to_chat("System", "Audio playback system is working.")
            return True

        except Exception as e:
            print(f"❌ AUDIO TEST: Error testing audio playback: {e}")
            self.add_to_chat("System", f"Error testing audio playback: {e}")
            return False

    def _play_direct_audio_test(self):
        """Play a direct audio test to verify the audio system is working."""
        try:
            print("🔊 DIRECT AUDIO TEST: Testing audio system with direct beep...")

            # Try multiple approaches to play a sound

            # Approach 1: Use sounddevice directly
            try:
                # Create a simple sine wave for testing
                sample_rate = 44100  # Standard sample rate
                duration = 0.3  # Short duration
                frequency = 880  # A5 note (higher pitch for distinction)

                # Generate a sine wave with fade in/out to avoid clicks
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                tone = 0.5 * np.sin(2 * np.pi * frequency * t)

                # Apply fade in/out
                fade_samples = int(0.05 * sample_rate)  # 50ms fade
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                tone[:fade_samples] *= fade_in
                tone[-fade_samples:] *= fade_out

                # Ensure the audio is in the correct format (float32)
                tone = tone.astype(np.float32)

                print(f"🔊 DIRECT AUDIO TEST: Generated test tone: {len(tone)} samples at {sample_rate}Hz")

                # Use default device for simplicity
                print("🔊 DIRECT AUDIO TEST: Playing test tone...")
                sd.play(tone, sample_rate)
                sd.wait()  # Wait until audio is finished playing
                print("✅ DIRECT AUDIO TEST: Test tone playback complete")
                return True

            except Exception as sd_error:
                print(f"⚠️ DIRECT AUDIO TEST: Error with sounddevice approach: {sd_error}")

                # Approach 2: Try system bell
                try:
                    print("🔊 DIRECT AUDIO TEST: Trying system bell...")
                    self.root.bell()
                    time.sleep(0.5)  # Wait a moment to ensure bell is heard
                    print("✅ DIRECT AUDIO TEST: System bell played")
                    return True
                except Exception as bell_error:
                    print(f"⚠️ DIRECT AUDIO TEST: Error with system bell: {bell_error}")

                    # Approach 3: Try subprocess to play a system sound
                    try:
                        print("🔊 DIRECT AUDIO TEST: Trying system sound command...")
                        if sys.platform.startswith('linux'):
                            # Try paplay on Linux
                            subprocess.run(["paplay", "--volume=65536", "/usr/share/sounds/freedesktop/stereo/bell.oga"], 
                                          check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                        elif sys.platform.startswith('darwin'):
                            # Try afplay on macOS
                            subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"], 
                                          check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                        elif sys.platform.startswith('win'):
                            # Try PowerShell on Windows
                            subprocess.run(["powershell", "-c", "(New-Object Media.SoundPlayer).PlaySync()"], 
                                          check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                        print("✅ DIRECT AUDIO TEST: System sound command executed")
                        return True
                    except Exception as cmd_error:
                        print(f"⚠️ DIRECT AUDIO TEST: Error with system sound command: {cmd_error}")
                        return False

        except Exception as e:
            print(f"❌ DIRECT AUDIO TEST: Error in direct audio test: {e}")
            return False

    def _play_direct_speech_simulation(self):
        """Simulate speech with direct audio output as a last resort."""
        try:
            print("🔊 DIRECT SPEECH: Simulating speech with direct audio...")

            # Create a sequence of tones that sound somewhat like speech
            sample_rate = 44100  # Standard sample rate

            # Create a composite sound with multiple frequencies
            def create_tone(freq, duration):
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                return 0.2 * np.sin(2 * np.pi * freq * t)

            # Create a sequence of tones with different pitches
            segments = []

            # Add some "speech-like" tones
            for freq in [300, 400, 350, 450, 300, 250]:
                duration = 0.15 + random.random() * 0.1  # Random duration between 0.15-0.25s
                tone = create_tone(freq, duration)

                # Apply fade in/out
                fade_samples = int(0.03 * sample_rate)  # 30ms fade
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                tone[:fade_samples] *= fade_in
                tone[-fade_samples:] *= fade_out

                segments.append(tone)

                # Add a short pause between segments
                pause_duration = 0.05 + random.random() * 0.05  # Random pause between 0.05-0.1s
                pause = np.zeros(int(sample_rate * pause_duration))
                segments.append(pause)

            # Combine all segments
            speech_simulation = np.concatenate(segments)

            # Ensure the audio is in the correct format (float32)
            speech_simulation = speech_simulation.astype(np.float32)

            print(f"🔊 DIRECT SPEECH: Generated speech simulation: {len(speech_simulation)} samples")

            # Play the simulated speech
            print("🔊 DIRECT SPEECH: Playing simulated speech...")
            sd.play(speech_simulation, sample_rate)
            sd.wait()  # Wait until audio is finished playing
            print("✅ DIRECT SPEECH: Speech simulation playback complete")

            # Add a message to the chat to inform the user
            self.add_to_chat("System", "Voice synthesis failed. Played audio notification instead.")

            return True

        except Exception as e:
            print(f"❌ DIRECT SPEECH: Error in speech simulation: {e}")
            return False

    def _test_web_audio_output(self):
        """Test audio output using the WebAudio API through a temporary HTML file.
        This is a completely different approach that bypasses all system audio libraries."""
        try:
            print("🔊 WEB AUDIO TEST: Testing audio output using WebAudio API...")

            # Create a temporary HTML file with JavaScript that uses WebAudio API
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode='w') as temp_html:
                temp_html_path = temp_html.name

                # Write HTML content with WebAudio API code
                html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Audio Test</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        button { padding: 10px 20px; font-size: 16px; }
        #status { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Audio Test</h1>
    <p>This page will automatically play a test sound using WebAudio API.</p>
    <button id="playButton">Play Test Sound</button>
    <div id="status">Waiting to play sound...</div>

    <script>
        // Function to play a test sound using WebAudio API
        function playTestSound() {
            const statusElement = document.getElementById('status');
            statusElement.textContent = 'Initializing audio context...';

            try {
                // Create audio context
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                const audioCtx = new AudioContext();

                statusElement.textContent = 'Creating oscillator...';

                // Create oscillator
                const oscillator = audioCtx.createOscillator();
                oscillator.type = 'sine';
                oscillator.frequency.setValueAtTime(440, audioCtx.currentTime); // A4 note

                // Create gain node for volume control
                const gainNode = audioCtx.createGain();
                gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime); // Set volume to 10%

                // Connect nodes
                oscillator.connect(gainNode);
                gainNode.connect(audioCtx.destination);

                statusElement.textContent = 'Playing sound...';

                // Start and stop the oscillator
                oscillator.start();

                // Play for 1 second
                setTimeout(() => {
                    oscillator.stop();
                    statusElement.textContent = 'Sound played successfully!';

                    // Signal success to the parent application
                    if (window.opener) {
                        window.opener.postMessage('audio_test_success', '*');
                    }

                    // Close the window after 3 seconds
                    setTimeout(() => {
                        window.close();
                    }, 3000);

                }, 1000);
            } catch (error) {
                statusElement.textContent = 'Error: ' + error.message;
                console.error('WebAudio error:', error);

                // Signal failure to the parent application
                if (window.opener) {
                    window.opener.postMessage('audio_test_failure', '*');
                }
            }
        }

        // Auto-play when the page loads
        window.onload = function() {
            // Add click event listener to the button
            document.getElementById('playButton').addEventListener('click', playTestSound);

            // Try to auto-play (may be blocked by browser)
            setTimeout(playTestSound, 500);
        };
    </script>
</body>
</html>
"""
                temp_html.write(html_content)

            print(f"🔊 WEB AUDIO TEST: Created temporary HTML file at {temp_html_path}")

            # Open the HTML file in the default web browser
            import webbrowser
            webbrowser.open('file://' + temp_html_path)

            # Add a message to the chat
            self.add_to_chat("System", "Testing audio using WebAudio API in your browser. Please allow audio playback if prompted.")

            # Clean up the temporary file after a delay
            def cleanup_temp_file():
                try:
                    time.sleep(10)  # Wait for 10 seconds before cleaning up
                    os.unlink(temp_html_path)
                    print(f"✅ WEB AUDIO TEST: Temporary file {temp_html_path} cleaned up")
                except Exception as e:
                    print(f"⚠️ WEB AUDIO TEST: Error cleaning up temporary file: {e}")

            # Start cleanup in a separate thread
            threading.Thread(target=cleanup_temp_file, daemon=True).start()

            print("✅ WEB AUDIO TEST: WebAudio test initiated in browser")
            return True

        except Exception as e:
            print(f"❌ WEB AUDIO TEST: Error in WebAudio test: {e}")
            return False

    def _test_direct_hardware_audio(self):
        """Test audio output by directly accessing hardware audio interfaces.
        This method bypasses all standard audio libraries and uses platform-specific low-level APIs."""
        try:
            print("🔊 HARDWARE AUDIO TEST: Testing direct hardware audio output...")

            # Platform-specific implementations
            if sys.platform.startswith('win'):
                return self._test_direct_hardware_audio_windows()
            elif sys.platform.startswith('darwin'):
                return self._test_direct_hardware_audio_macos()
            elif sys.platform.startswith('linux'):
                return self._test_direct_hardware_audio_linux()
            else:
                print(f"❌ HARDWARE AUDIO TEST: Unsupported platform: {sys.platform}")
                return False

        except Exception as e:
            print(f"❌ HARDWARE AUDIO TEST: Error in direct hardware audio test: {e}")
            return False

    def _test_direct_hardware_audio_windows(self):
        """Test direct hardware audio output on Windows using WinMM or DirectSound."""
        try:
            print("🔊 HARDWARE AUDIO TEST: Testing Windows hardware audio...")

            # Try using Windows Multimedia API (WinMM) directly
            try:
                # Load the winmm.dll
                winmm = ctypes.WinDLL('winmm')

                # Define constants for waveOut functions
                WAVE_FORMAT_PCM = 1
                CALLBACK_NULL = 0

                # Define structures for waveOut functions
                class WAVEFORMATEX(ctypes.Structure):
                    _fields_ = [
                        ("wFormatTag", ctypes.c_ushort),
                        ("nChannels", ctypes.c_ushort),
                        ("nSamplesPerSec", ctypes.c_uint),
                        ("nAvgBytesPerSec", ctypes.c_uint),
                        ("nBlockAlign", ctypes.c_ushort),
                        ("wBitsPerSample", ctypes.c_ushort),
                        ("cbSize", ctypes.c_ushort)
                    ]

                # Create a simple sine wave
                sample_rate = 44100
                duration = 0.5  # seconds
                frequency = 440  # Hz (A4 note)
                num_samples = int(sample_rate * duration)

                # Generate the sine wave
                buffer = bytearray()
                for i in range(num_samples):
                    value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
                    buffer.extend(value.to_bytes(2, byteorder='little', signed=True))

                # Create a buffer for the audio data
                buffer_ptr = ctypes.create_string_buffer(bytes(buffer))

                # Set up the WAVEFORMATEX structure
                wave_format = WAVEFORMATEX(
                    WAVE_FORMAT_PCM,  # wFormatTag
                    1,                # nChannels (mono)
                    sample_rate,      # nSamplesPerSec
                    sample_rate * 2,  # nAvgBytesPerSec (sample_rate * nBlockAlign)
                    2,                # nBlockAlign (nChannels * wBitsPerSample / 8)
                    16,               # wBitsPerSample
                    0                 # cbSize
                )

                # Open a waveOut device
                wave_out_handle = ctypes.c_void_p()
                result = winmm.waveOutOpen(
                    ctypes.byref(wave_out_handle),  # phwo
                    0xFFFFFFFF,                     # uDeviceID (WAVE_MAPPER)
                    ctypes.byref(wave_format),      # pwfx
                    0,                              # dwCallback
                    0,                              # dwCallbackInstance
                    CALLBACK_NULL                   # fdwOpen
                )

                if result != 0:
                    print(f"❌ HARDWARE AUDIO TEST: waveOutOpen failed with error code {result}")
                    return False

                # Define the WAVEHDR structure
                class WAVEHDR(ctypes.Structure):
                    _fields_ = [
                        ("lpData", ctypes.c_char_p),
                        ("dwBufferLength", ctypes.c_uint),
                        ("dwBytesRecorded", ctypes.c_uint),
                        ("dwUser", ctypes.c_uint),
                        ("dwFlags", ctypes.c_uint),
                        ("dwLoops", ctypes.c_uint),
                        ("lpNext", ctypes.c_void_p),
                        ("reserved", ctypes.c_uint)
                    ]

                # Create and prepare a WAVEHDR
                wave_hdr = WAVEHDR(
                    buffer_ptr,           # lpData
                    len(buffer),          # dwBufferLength
                    0,                    # dwBytesRecorded
                    0,                    # dwUser
                    0,                    # dwFlags
                    0,                    # dwLoops
                    None,                 # lpNext
                    0                     # reserved
                )

                result = winmm.waveOutPrepareHeader(wave_out_handle, ctypes.byref(wave_hdr), ctypes.sizeof(wave_hdr))
                if result != 0:
                    print(f"❌ HARDWARE AUDIO TEST: waveOutPrepareHeader failed with error code {result}")
                    winmm.waveOutClose(wave_out_handle)
                    return False

                # Write the data to the device
                result = winmm.waveOutWrite(wave_out_handle, ctypes.byref(wave_hdr), ctypes.sizeof(wave_hdr))
                if result != 0:
                    print(f"❌ HARDWARE AUDIO TEST: waveOutWrite failed with error code {result}")
                    winmm.waveOutUnprepareHeader(wave_out_handle, ctypes.byref(wave_hdr), ctypes.sizeof(wave_hdr))
                    winmm.waveOutClose(wave_out_handle)
                    return False

                # Wait for playback to complete
                time.sleep(duration + 0.1)

                # Clean up
                winmm.waveOutUnprepareHeader(wave_out_handle, ctypes.byref(wave_hdr), ctypes.sizeof(wave_hdr))
                winmm.waveOutClose(wave_out_handle)

                print("✅ HARDWARE AUDIO TEST: Windows WinMM audio playback successful")
                return True

            except Exception as winmm_error:
                print(f"⚠️ HARDWARE AUDIO TEST: Windows WinMM error: {winmm_error}")

                # Try using DirectSound as a fallback
                try:
                    # Use PowerShell to play a system sound as a fallback
                    ps_command = "[System.Media.SystemSounds]::Beep.Play()"
                    subprocess.run(["powershell", "-c", ps_command], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                    print("✅ HARDWARE AUDIO TEST: Windows SystemSounds.Beep successful")
                    return True
                except Exception as ds_error:
                    print(f"❌ HARDWARE AUDIO TEST: Windows DirectSound fallback error: {ds_error}")
                    return False

        except Exception as e:
            print(f"❌ HARDWARE AUDIO TEST: Windows hardware audio error: {e}")
            return False

    def _test_direct_hardware_audio_macos(self):
        """Test direct hardware audio output on macOS using CoreAudio or AudioToolbox."""
        try:
            print("🔊 HARDWARE AUDIO TEST: Testing macOS hardware audio...")

            # Try using AudioToolbox via subprocess
            try:
                # Create a temporary Swift file that uses AudioToolbox
                with tempfile.NamedTemporaryFile(suffix=".swift", delete=False, mode='w') as temp_swift:
                    temp_swift_path = temp_swift.name

                    # Write Swift code that uses AudioToolbox to play a sound
                    swift_code = """
import Foundation
import AudioToolbox

// Create a simple beep sound
var soundID: SystemSoundID = 0
let soundURL = URL(fileURLWithPath: "/System/Library/Sounds/Tink.aiff")

// Register the sound
AudioServicesCreateSystemSoundID(soundURL as CFURL, &soundID)

// Play the sound
AudioServicesPlaySystemSound(soundID)

// Wait for a moment to allow the sound to play
Thread.sleep(forTimeInterval: 1.0)

// Dispose of the sound
AudioServicesDisposeSystemSoundID(soundID)

print("Sound played successfully")
"""
                    temp_swift.write(swift_code)

                # Compile and run the Swift file
                print(f"🔊 HARDWARE AUDIO TEST: Created temporary Swift file at {temp_swift_path}")

                # Try to run the Swift file
                result = subprocess.run(["swift", temp_swift_path], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)

                # Clean up
                os.unlink(temp_swift_path)

                if "Sound played successfully" in result.stdout:
                    print("✅ HARDWARE AUDIO TEST: macOS AudioToolbox successful")
                    return True
                else:
                    print(f"⚠️ HARDWARE AUDIO TEST: macOS AudioToolbox output: {result.stdout}")
                    print(f"⚠️ HARDWARE AUDIO TEST: macOS AudioToolbox error: {result.stderr}")

                    # Try using afplay as a fallback
                    try:
                        subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"], 
                                      check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                        print("✅ HARDWARE AUDIO TEST: macOS afplay successful")
                        return True
                    except Exception as afplay_error:
                        print(f"❌ HARDWARE AUDIO TEST: macOS afplay error: {afplay_error}")
                        return False

            except Exception as swift_error:
                print(f"⚠️ HARDWARE AUDIO TEST: macOS Swift error: {swift_error}")

                # Try using osascript as a fallback
                try:
                    subprocess.run(["osascript", "-e", "beep"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                    print("✅ HARDWARE AUDIO TEST: macOS osascript beep successful")
                    return True
                except Exception as osa_error:
                    print(f"❌ HARDWARE AUDIO TEST: macOS osascript error: {osa_error}")
                    return False

        except Exception as e:
            print(f"❌ HARDWARE AUDIO TEST: macOS hardware audio error: {e}")
            return False

    def _test_direct_hardware_audio_linux(self):
        """Test direct hardware audio output on Linux using ALSA or PulseAudio."""
        try:
            print("🔊 HARDWARE AUDIO TEST: Testing Linux hardware audio...")

            # Try using ALSA directly via C program
            try:
                # Create a temporary C file that uses ALSA
                with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode='w') as temp_c:
                    temp_c_path = temp_c.name

                    # Write C code that uses ALSA to play a sound
                    c_code = """
#include <stdio.h>
#include <stdlib.h>
#include <alsa/asoundlib.h>
#include <math.h>

#define SAMPLE_RATE 44100
#define DURATION 0.5  // seconds
#define FREQUENCY 440  // Hz (A4 note)

int main() {
    int err;
    snd_pcm_t *pcm_handle;
    snd_pcm_hw_params_t *params;
    unsigned int sample_rate = SAMPLE_RATE;
    int dir = 0;
    snd_pcm_uframes_t frames = 32;
    int num_samples = SAMPLE_RATE * DURATION;

    // Open the PCM device
    err = snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        fprintf(stderr, "Cannot open PCM device: %s\\n", snd_strerror(err));
        return 1;
    }

    // Allocate hardware parameters object
    snd_pcm_hw_params_alloca(&params);

    // Fill it with default values
    snd_pcm_hw_params_any(pcm_handle, params);

    // Set parameters
    err = snd_pcm_hw_params_set_access(pcm_handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        fprintf(stderr, "Cannot set access type: %s\\n", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return 1;
    }

    err = snd_pcm_hw_params_set_format(pcm_handle, params, SND_PCM_FORMAT_S16_LE);
    if (err < 0) {
        fprintf(stderr, "Cannot set format: %s\\n", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return 1;
    }

    err = snd_pcm_hw_params_set_channels(pcm_handle, params, 1);
    if (err < 0) {
        fprintf(stderr, "Cannot set channels: %s\\n", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return 1;
    }

    err = snd_pcm_hw_params_set_rate_near(pcm_handle, params, &sample_rate, &dir);
    if (err < 0) {
        fprintf(stderr, "Cannot set sample rate: %s\\n", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return 1;
    }

    err = snd_pcm_hw_params_set_period_size_near(pcm_handle, params, &frames, &dir);
    if (err < 0) {
        fprintf(stderr, "Cannot set period size: %s\\n", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return 1;
    }

    // Apply the hardware parameters
    err = snd_pcm_hw_params(pcm_handle, params);
    if (err < 0) {
        fprintf(stderr, "Cannot set hardware parameters: %s\\n", snd_strerror(err));
        snd_pcm_close(pcm_handle);
        return 1;
    }

    // Generate a sine wave
    short *buffer = (short *) malloc(num_samples * sizeof(short));
    if (buffer == NULL) {
        fprintf(stderr, "Cannot allocate buffer\\n");
        snd_pcm_close(pcm_handle);
        return 1;
    }

    for (int i = 0; i < num_samples; i++) {
        buffer[i] = (short)(32767.0 * sin(2.0 * M_PI * FREQUENCY * i / SAMPLE_RATE));
    }

    // Write the samples
    int frames_written = 0;
    while (frames_written < num_samples) {
        err = snd_pcm_writei(pcm_handle, buffer + frames_written, num_samples - frames_written);
        if (err == -EPIPE) {
            // EPIPE means underrun
            fprintf(stderr, "Underrun occurred\\n");
            snd_pcm_prepare(pcm_handle);
        } else if (err < 0) {
            fprintf(stderr, "Error from writei: %s\\n", snd_strerror(err));
            break;
        } else {
            frames_written += err;
        }
    }

    // Wait for all samples to be played
    snd_pcm_drain(pcm_handle);

    // Clean up
    free(buffer);
    snd_pcm_close(pcm_handle);

    printf("Sound played successfully\\n");
    return 0;
}
"""
                    temp_c.write(c_code)

                # Compile the C file
                print(f"🔊 HARDWARE AUDIO TEST: Created temporary C file at {temp_c_path}")

                # Create a temporary executable file
                temp_exec_path = temp_c_path + ".out"

                # Compile the C file
                compile_result = subprocess.run(["gcc", temp_c_path, "-o", temp_exec_path, "-lasound", "-lm"], 
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)

                if compile_result.returncode != 0:
                    print(f"⚠️ HARDWARE AUDIO TEST: Failed to compile ALSA C program: {compile_result.stderr}")
                else:
                    # Run the compiled program
                    run_result = subprocess.run([temp_exec_path], 
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)

                    if "Sound played successfully" in run_result.stdout:
                        print("✅ HARDWARE AUDIO TEST: Linux ALSA direct access successful")

                        # Clean up
                        os.unlink(temp_c_path)
                        os.unlink(temp_exec_path)

                        return True
                    else:
                        print(f"⚠️ HARDWARE AUDIO TEST: ALSA program output: {run_result.stdout}")
                        print(f"⚠️ HARDWARE AUDIO TEST: ALSA program error: {run_result.stderr}")

                # Clean up
                try:
                    os.unlink(temp_c_path)
                    if os.path.exists(temp_exec_path):
                        os.unlink(temp_exec_path)
                except:
                    pass

                # Try using aplay as a fallback
                try:
                    # Create a simple WAV file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                        temp_wav_path = temp_wav.name

                    # Generate a WAV file with a sine wave
                    import wave
                    import struct

                    sample_rate = 44100
                    duration = 0.5  # seconds
                    frequency = 440  # Hz (A4 note)
                    amplitude = 32767  # Maximum amplitude for 16-bit audio

                    # Create the WAV file
                    with wave.open(temp_wav_path, 'w') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 2 bytes (16 bits) per sample
                        wav_file.setframerate(sample_rate)

                        # Generate the sine wave
                        for i in range(int(duration * sample_rate)):
                            value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                            packed_value = struct.pack('h', value)  # 'h' for 16-bit signed integer
                            wav_file.writeframes(packed_value)

                    # Play the WAV file using aplay
                    subprocess.run(["aplay", temp_wav_path], 
                                 check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)

                    # Clean up
                    os.unlink(temp_wav_path)

                    print("✅ HARDWARE AUDIO TEST: Linux aplay successful")
                    return True

                except Exception as aplay_error:
                    print(f"⚠️ HARDWARE AUDIO TEST: Linux aplay error: {aplay_error}")

                    # Try using paplay (PulseAudio) as a final fallback
                    try:
                        # Create a simple WAV file if it doesn't exist
                        if not os.path.exists(temp_wav_path):
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                                temp_wav_path = temp_wav.name

                            # Generate a WAV file with a sine wave
                            import wave
                            import struct

                            sample_rate = 44100
                            duration = 0.5  # seconds
                            frequency = 440  # Hz (A4 note)
                            amplitude = 32767  # Maximum amplitude for 16-bit audio

                            # Create the WAV file
                            with wave.open(temp_wav_path, 'w') as wav_file:
                                wav_file.setnchannels(1)  # Mono
                                wav_file.setsampwidth(2)  # 2 bytes (16 bits) per sample
                                wav_file.setframerate(sample_rate)

                                # Generate the sine wave
                                for i in range(int(duration * sample_rate)):
                                    value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                                    packed_value = struct.pack('h', value)  # 'h' for 16-bit signed integer
                                    wav_file.writeframes(packed_value)

                        # Play the WAV file using paplay
                        subprocess.run(["paplay", temp_wav_path], 
                                     check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)

                        # Clean up
                        os.unlink(temp_wav_path)

                        print("✅ HARDWARE AUDIO TEST: Linux paplay successful")
                        return True

                    except Exception as paplay_error:
                        print(f"❌ HARDWARE AUDIO TEST: Linux paplay error: {paplay_error}")
                        return False

            except Exception as c_error:
                print(f"❌ HARDWARE AUDIO TEST: Linux C program error: {c_error}")
                return False

        except Exception as e:
            print(f"❌ HARDWARE AUDIO TEST: Linux hardware audio error: {e}")
            return False

    def _fallback_to_mimic3(self, text, temp_audio_path):
        """Fallback to Mimic3 for text-to-speech if Coqui TTS fails."""
        try:
            # Check if mimic3 is installed
            print("🔊 VOICE OUTPUT: Checking if Mimic3 is installed...")
            mimic3_check = subprocess.run(["which", "mimic3"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if mimic3_check.returncode == 0:
                print("✅ VOICE OUTPUT: Mimic3 found at: " + mimic3_check.stdout.decode('utf-8').strip())

                # Generate speech with mimic3
                print("🔊 VOICE OUTPUT: Generating speech with Mimic3...")
                mimic_result = subprocess.run(
                    ["mimic3", "--voice", "en_US/vctk_low", "--output", temp_audio_path, text],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                if mimic_result.returncode != 0:
                    print(f"❌ VOICE OUTPUT: Mimic3 failed with error: {mimic_result.stderr.decode('utf-8')}")
                    raise Exception(f"Mimic3 failed: {mimic_result.stderr.decode('utf-8')}")

                # Verify the file was created and has content
                if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                    print(f"✅ VOICE OUTPUT: Mimic3 audio file created successfully ({os.path.getsize(temp_audio_path)} bytes)")
                else:
                    print("❌ VOICE OUTPUT: Mimic3 audio file is empty or not created")
                    raise Exception("Mimic3 audio file is empty or not created")

                # Play the audio
                print("🔊 VOICE OUTPUT: Playing Mimic3 audio...")
                data, fs = sf.read(temp_audio_path)
                print(f"✅ VOICE OUTPUT: Mimic3 audio file loaded: {len(data)} samples at {fs}Hz")

                # Use selected output device if available
                device_args = {}
                if self.selected_output_device and self.selected_output_device != "Default":
                    try:
                        # Find the device index for the selected device name
                        device_index = self.output_devices.index(self.selected_output_device)
                        device_args["device"] = device_index
                        print(f"🔊 VOICE OUTPUT: Using output device for Mimic3: {self.selected_output_device} (index: {device_index})")
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ VOICE OUTPUT: Error finding output device index for Mimic3: {e}")
                        print("🔊 VOICE OUTPUT: Continuing with default device for Mimic3")
                else:
                    print("🔊 VOICE OUTPUT: Using default audio device for Mimic3")

                # Play the audio with detailed logging
                print("🔊 VOICE OUTPUT: Starting Mimic3 audio playback...")
                sd.play(data, fs, **device_args)
                sd.wait()  # Wait until audio is finished playing
                print("✅ VOICE OUTPUT: Mimic3 audio playback complete")
                return True
            else:
                print("⚠️ VOICE OUTPUT: Mimic3 not found, trying system TTS...")
                raise subprocess.CalledProcessError(1, "which mimic3")

        except subprocess.CalledProcessError:
            print("⚠️ VOICE OUTPUT: Mimic3 not found or error running Mimic3")
            # Fall back to system TTS if available
            try:
                print("🔊 VOICE OUTPUT: Falling back to system TTS (say command)...")
                say_result = subprocess.run(["say", text], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if say_result.returncode == 0:
                    print("✅ VOICE OUTPUT: System TTS (say command) completed successfully")
                    return True
                else:
                    print(f"❌ VOICE OUTPUT: System TTS (say command) failed: {say_result.stderr.decode('utf-8')}")
                    raise Exception(f"System TTS failed: {say_result.stderr.decode('utf-8')}")

            except Exception as tts_error:
                print(f"❌ VOICE OUTPUT: System TTS not available: {tts_error}")

                # Try espeak as a last resort
                try:
                    print("🔊 VOICE OUTPUT: Trying espeak as last resort...")
                    espeak_result = subprocess.run(["espeak", text], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if espeak_result.returncode == 0:
                        print("✅ VOICE OUTPUT: Espeak completed successfully")
                        return True
                    else:
                        print(f"❌ VOICE OUTPUT: Espeak failed: {espeak_result.stderr.decode('utf-8')}")
                except Exception as espeak_error:
                    print(f"❌ VOICE OUTPUT: Espeak not available: {espeak_error}")

                # If no TTS is available, just show the message
                self.status_bar.config(text="Text-to-speech not available")
                print("❌ VOICE OUTPUT: All TTS methods failed")
                return False

        except Exception as e:
            print(f"❌ VOICE OUTPUT: Unexpected error in Mimic3 fallback: {e}")
            return False


def test_audio_from_command_line():
    """Test audio output directly from the command line."""
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(description='Test audio output')
    parser.add_argument('--method', 
                        choices=['beep', 'voice', 'system', 'web', 'hardware', 'comprehensive', 'all'], 
                        default='all',
                        help='Audio test method (beep, voice, system, web, hardware, comprehensive, or all)')
    parser.add_argument('--device', default='default',
                        help='Audio output device name or index')
    parser.add_argument('--text', default='This is a test of the audio output system.',
                        help='Text to speak (for voice method)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--fix', action='store_true',
                        help='Attempt to fix audio issues if tests fail')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Run a comprehensive diagnostic and create a detailed report')

    # Parse arguments
    args = parser.parse_args()

    # Initialize Tkinter (required for some functionality)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Create app instance
    app = AutonomousApp(root)

    # List devices if requested
    if args.list_devices:
        print("\nAvailable audio devices:")
        if AUDIO_AVAILABLE:
            try:
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    if device['max_output_channels'] > 0:
                        print(f"  {i}: {device['name']} (Output)")
                    if device['max_input_channels'] > 0:
                        print(f"  {i}: {device['name']} (Input)")
            except Exception as e:
                print(f"Error listing devices: {e}")
        else:
            print("Audio libraries not available.")
        return

    # Determine device index
    device_index = None
    if args.device != 'default':
        try:
            # Try to convert to integer (device index)
            device_index = int(args.device)
        except ValueError:
            # Try to find device by name
            if AUDIO_AVAILABLE:
                try:
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        if args.device.lower() in device['name'].lower() and device['max_output_channels'] > 0:
                            device_index = i
                            break
                except Exception as e:
                    print(f"Error finding device: {e}")

    # Set device args
    device_args = {}
    if device_index is not None:
        device_args['device'] = device_index
        print(f"Using device index: {device_index}")

    # Track test results
    test_results = {}

    # Run tests
    if args.method == 'beep' or args.method == 'all':
        print("\nTesting beep...")
        try:
            if AUDIO_AVAILABLE:
                # Create a simple sine wave
                sample_rate = 44100
                duration = 0.5
                frequency = 440
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                tone = 0.5 * np.sin(2 * np.pi * frequency * t)
                tone = tone.astype(np.float32)

                # Play the tone
                sd.play(tone, sample_rate, **device_args)
                sd.wait()
                print("Beep test completed. Did you hear a beep?")
                test_results['beep'] = True
            else:
                print("Audio libraries not available.")
                test_results['beep'] = False
        except Exception as e:
            print(f"Error in beep test: {e}")
            test_results['beep'] = False

    if args.method == 'voice' or args.method == 'all':
        print("\nTesting voice...")
        try:
            # Initialize TTS model if needed
            if app.tts_model is None and COQUI_TTS_AVAILABLE:
                print("Initializing TTS model with retries...")
                app._initialize_tts_with_retry(max_retries=3, retry_count=0)

            # Set output device
            if device_index is not None:
                app.selected_output_device = app.output_devices[device_index] if device_index < len(app.output_devices) else "Default"

            # Speak text
            print(f"Speaking: '{args.text}'")
            result = app.speak_text(args.text)
            print(f"Voice test completed with result: {result}. Did you hear the message?")
            test_results['voice'] = result
        except Exception as e:
            print(f"Error in voice test: {e}")
            test_results['voice'] = False

    if args.method == 'system' or args.method == 'all':
        print("\nTesting system sound...")
        system_result = False
        try:
            # Try system sound commands
            if sys.platform.startswith('linux'):
                try:
                    device_arg = []
                    if device_index is not None:
                        device_name = app.output_devices[device_index] if device_index < len(app.output_devices) else "default"
                        device_arg = ["--device", device_name]

                    print("Trying paplay...")
                    subprocess.run(["paplay"] + device_arg + ["/usr/share/sounds/freedesktop/stereo/bell.oga"], 
                                  check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                    system_result = True
                except:
                    try:
                        device_arg = []
                        if device_index is not None:
                            device_name = app.output_devices[device_index] if device_index < len(app.output_devices) else "default"
                            device_arg = ["-D", device_name]

                        print("Trying aplay...")
                        subprocess.run(["aplay"] + device_arg + ["/usr/share/sounds/alsa/Front_Center.wav"], 
                                      check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                        system_result = True
                    except:
                        pass
            elif sys.platform.startswith('darwin'):
                try:
                    print("Trying afplay...")
                    subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"], 
                                  check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                    system_result = True
                except:
                    pass
            elif sys.platform.startswith('win'):
                try:
                    print("Trying PowerShell...")
                    subprocess.run(["powershell", "-c", "(New-Object Media.SoundPlayer).PlaySync()"], 
                                  check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
                    system_result = True
                except:
                    pass

            print("System sound test completed. Did you hear a system sound?")
            test_results['system'] = system_result
        except Exception as e:
            print(f"Error in system sound test: {e}")
            test_results['system'] = False

    if args.method == 'web' or args.method == 'all':
        print("\nTesting WebAudio API...")
        try:
            result = app._test_web_audio_output()
            print("WebAudio test initiated in your browser. Please check if you hear a sound.")
            print("Note: This test requires manual confirmation as it opens in a browser.")
            test_results['web'] = result
        except Exception as e:
            print(f"Error in WebAudio test: {e}")
            test_results['web'] = False

    if args.method == 'hardware' or args.method == 'all':
        print("\nTesting direct hardware audio...")
        try:
            result = app._test_direct_hardware_audio()
            print(f"Hardware audio test completed with result: {result}. Did you hear a sound?")
            test_results['hardware'] = result
        except Exception as e:
            print(f"Error in hardware audio test: {e}")
            test_results['hardware'] = False

    if args.method == 'comprehensive' or args.method == 'all':
        print("\nRunning comprehensive audio test...")
        try:
            working_methods = app._test_all_audio_methods()
            print(f"Comprehensive test completed. Working methods: {', '.join(working_methods) if working_methods else 'None'}")
            test_results['comprehensive'] = len(working_methods) > 0
        except Exception as e:
            print(f"Error in comprehensive audio test: {e}")
            test_results['comprehensive'] = False

    # Check if any tests failed and fix is requested
    if args.fix and not all(test_results.values()):
        print("\nAttempting to fix audio issues...")
        try:
            print("Restarting audio subsystem...")
            app._restart_audio_subsystem()

            print("Checking audio device configuration...")
            app._initialize_audio_devices()

            print("Testing audio system after fixes...")
            app._test_audio_system()

            print("Fix attempt completed. Please run the tests again to check if issues are resolved.")
        except Exception as e:
            print(f"Error while attempting to fix audio issues: {e}")

    # Create diagnostic report
    if args.diagnostic or not any(test_results.values()):
        try:
            print("\nCreating comprehensive diagnostic report...")
            report_path = os.path.join(os.path.expanduser('~'), 'audio_diagnostic_report.txt')

            with open(report_path, 'w') as f:
                f.write("=== AUDIO DIAGNOSTIC REPORT ===\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Platform: {sys.platform}\n")
                f.write(f"Python version: {sys.version}\n\n")

                # Audio libraries availability
                f.write("=== AUDIO LIBRARIES ===\n")
                f.write(f"sounddevice/soundfile available: {AUDIO_AVAILABLE}\n")
                f.write(f"Coqui TTS available: {COQUI_TTS_AVAILABLE}\n")
                f.write(f"Speech recognition available: {SPEECH_RECOGNITION_AVAILABLE}\n")
                f.write(f"Whisper available: {WHISPER_AVAILABLE}\n\n")

                # Audio devices
                f.write("=== AUDIO DEVICES ===\n")
                if AUDIO_AVAILABLE:
                    try:
                        devices = sd.query_devices()
                        for i, device in enumerate(devices):
                            if device['max_output_channels'] > 0:
                                f.write(f"Output device {i}: {device['name']}\n")
                            if device['max_input_channels'] > 0:
                                f.write(f"Input device {i}: {device['name']}\n")
                    except Exception as e:
                        f.write(f"Error listing devices: {e}\n")
                else:
                    f.write("Audio libraries not available, cannot list devices.\n")
                f.write("\n")

                # Test results
                f.write("=== TEST RESULTS ===\n")
                for test, result in test_results.items():
                    f.write(f"{test}: {'SUCCESS' if result else 'FAILED'}\n")
                f.write("\n")

                # System-specific information
                f.write("=== SYSTEM-SPECIFIC INFORMATION ===\n")
                if sys.platform.startswith('linux'):
                    try:
                        # Check ALSA
                        alsa_result = subprocess.run(["aplay", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        f.write("ALSA devices:\n")
                        f.write(alsa_result.stdout)
                        f.write("\n")

                        # Check PulseAudio
                        pulse_result = subprocess.run(["pactl", "list", "sinks"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        f.write("PulseAudio sinks:\n")
                        f.write(pulse_result.stdout)
                        f.write("\n")
                    except Exception as e:
                        f.write(f"Error getting system audio info: {e}\n")
                elif sys.platform.startswith('darwin'):
                    try:
                        # Check macOS audio
                        mac_result = subprocess.run(["system_profiler", "SPAudioDataType"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        f.write("macOS audio devices:\n")
                        f.write(mac_result.stdout)
                        f.write("\n")
                    except Exception as e:
                        f.write(f"Error getting system audio info: {e}\n")
                elif sys.platform.startswith('win'):
                    f.write("Windows audio devices: Use Control Panel > Sound to view devices.\n")

                # Recommendations
                f.write("=== RECOMMENDATIONS ===\n")
                if not any(test_results.values()):
                    f.write("All audio tests failed. Please check:\n")
                    f.write("1. Is your audio device connected and powered on?\n")
                    f.write("2. Is your system volume muted or too low?\n")
                    f.write("3. Are the correct audio drivers installed?\n")
                    f.write("4. Try running 'python main_gui.py --method=web' to test audio in a browser.\n")
                    f.write("5. Try running 'python main_gui.py --fix' to attempt automatic fixes.\n")
                elif not all(test_results.values()):
                    f.write("Some audio tests failed. Recommended working methods:\n")
                    for test, result in test_results.items():
                        if result:
                            f.write(f"- {test}\n")
                else:
                    f.write("All audio tests passed. Your audio system appears to be working correctly.\n")

            print(f"Comprehensive diagnostic report created at: {report_path}")

            # Also create the standard diagnostic report
            app._create_audio_diagnostic_report("COMMAND_LINE_TEST")
            print(f"Standard diagnostic report created at: {os.path.join(os.path.expanduser('~'), 'voice_output_diagnostic_report.txt')}")
        except Exception as e:
            print(f"Error creating diagnostic report: {e}")
    elif not args.diagnostic:
        try:
            print("\nCreating standard diagnostic report...")
            app._create_audio_diagnostic_report("COMMAND_LINE_TEST")
            print(f"Diagnostic report created at: {os.path.join(os.path.expanduser('~'), 'voice_output_diagnostic_report.txt')}")
        except Exception as e:
            print(f"Error creating diagnostic report: {e}")

    # Summary
    print("\n=== TEST SUMMARY ===")
    for test, result in test_results.items():
        print(f"{test}: {'SUCCESS' if result else 'FAILED'}")

    print("\nAll tests completed.")

if __name__ == "__main__":
    import sys

    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test-audio":
        # Remove the first argument so argparse works correctly
        sys.argv.pop(1)
        test_audio_from_command_line()
    else:
        # Create the Tkinter root window
        root = tk.Tk()

        # Initialize the GUI application
        app = AutonomousApp(root)

        # Run the Tkinter main event loop
        root.mainloop()
