# macOS Installation Guide

This document provides instructions for installing and running Real-Time Voice Cloning on macOS (including Apple Silicon/ARM64).

## Overview

This guide covers the official macOS support (including Apple Silicon/ARM64) for the Real-Time Voice Cloning project. The implementation includes platform-specific configurations and improvements to ensure smooth installation and operation on macOS.

## Prerequisites

1. **Homebrew** - Install from https://brew.sh if not already installed
2. **Python 3.10** - Will be installed automatically by `uv`

## Installation Steps

### 1. Install System Dependencies

```bash
brew install ffmpeg libsndfile qt@5
```

### 2. Set Up Python Environment

The project uses `uv` for package management. Python 3.10 will be installed automatically.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project directory
cd Real-Time-Voice-Cloning

# Pin Python 3.10 (required for PyQt5 ARM64 wheels)
uv python install 3.10
uv python pin 3.10
```

### 3. Install Python Dependencies

```bash
# Set environment variables for Qt5 and libsndfile
export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"
export CMAKE_PREFIX_PATH="/opt/homebrew/opt/qt@5"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

# Install dependencies
uv sync --extra cpu
```

**Note:** If `uv sync` fails, you can install packages manually:
```bash
uv pip install inflect==5.3.0 librosa==0.9.2 matplotlib==3.5.1 "numpy>=1.26,<2" Pillow==8.4.0 scikit-learn==1.0.2 "scipy>=1.7" "setuptools<=80.8.0" sounddevice==0.4.3 soundfile==0.10.3.post1 tqdm==4.62.3 umap-learn==0.5.2 Unidecode==1.3.2 urllib3==1.26.7 visdom==0.1.8.9 webrtcvad==2.0.10
uv pip install --only-binary :all: "PyQt5>=5.15.6,<5.16"
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch>=1.13,<1.14"
```

### 4. Run the Toolbox

```bash
# Set required environment variables
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

# Run the GUI toolbox
uv run --extra cpu demo_toolbox.py

# Or run the CLI version
uv run --extra cpu demo_cli.py --help

# Or use the helper script (recommended)
./run_macos        # GUI version
./run_macos cli    # CLI version
```

## Key Differences from Linux/Windows

1. **Python Version**: Uses Python 3.10 instead of 3.9 (required for PyQt5 ARM64 wheels)
2. **PyQt5 Version**: Uses PyQt5 5.15.11+ (has ARM64 wheels) instead of 5.15.6
3. **PyTorch Version**: Uses PyTorch 1.13+ (supports Python 3.10) instead of 1.10
4. **Library Path**: Requires `DYLD_LIBRARY_PATH` to be set for `libsndfile`

## Troubleshooting

### Issue: `sndfile library not found`
**Solution:** Ensure `DYLD_LIBRARY_PATH` includes `/opt/homebrew/lib`:
```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
```

### Issue: Model download fails (especially synthesizer.pt)
**Solution:** The download function has been improved to handle Google Drive's virus scan confirmation page for large files. If downloads still fail:
1. Check your internet connection
2. Try downloading models manually from the [Google Drive folder](https://drive.google.com/drive/folders/1fU6umc5uQAVR2udZdHX-lDgXYzTyqG_j)
3. Place them in `saved_models/default/` directory

### Issue: PyQt5 build fails
**Solution:** Ensure Qt5 is installed and environment variables are set:
```bash
brew install qt@5
export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"
export CMAKE_PREFIX_PATH="/opt/homebrew/opt/qt@5"
```

### Issue: PyTorch installation fails
**Solution:** Use the PyTorch CPU index explicitly:
```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch>=1.13,<1.14"
```

## Technical Details

### Key Challenges Solved

1. **PyQt5 ARM64 Wheels**
   - Problem: PyQt5 5.15.6 doesn't have ARM64 wheels, requiring long build times
   - Solution: Use PyQt5 5.15.11+ which has pre-built ARM64 wheels

2. **PyTorch Python 3.10 Support**
   - Problem: PyTorch 1.10.* doesn't support Python 3.10
   - Solution: Use PyTorch 1.13+ for macOS which supports Python 3.10

3. **Library Path Issues**
   - Problem: `libsndfile` not found by Python's `soundfile` package
   - Solution: Set `DYLD_LIBRARY_PATH` to include `/opt/homebrew/lib`

4. **Qt5 Build Dependencies**
   - Problem: PyQt5 needs Qt5 development libraries
   - Solution: Install `qt@5` via Homebrew and set `CMAKE_PREFIX_PATH`

### Configuration Changes Made

The following changes were made to `pyproject.toml` to support macOS:

1. **Python Version**: Updated from `>=3.9,<3.10` to `>=3.10,<3.11`
   - Reason: PyQt5 5.15.6+ requires Python 3.10+ for ARM64 wheel support
2. **PyQt5 Version**: Changed from `==5.15.6` to `>=5.15.6,<5.16`
   - Reason: PyQt5 5.15.11 has pre-built ARM64 wheels, avoiding long build times
3. **PyTorch Version**: Added platform-specific requirements
   - Linux/Windows: `torch==1.10.*` (original)
   - macOS: `torch>=1.13,<1.14` (supports Python 3.10)
4. **Platform Support**: Added `sys_platform == 'darwin'` to `required-environments`

### Additional Improvements

#### Google Drive Download Fix
- Enhanced the download function in `utils/default_models.py` to properly handle Google Drive's virus scan confirmation page
- Large files (like synthesizer.pt at 370MB) now download correctly
- Added proper cookie handling and URL extraction from confirmation pages

#### Helper Script (`run_macos`)
- Automated setup script for macOS users
- Checks and installs system dependencies
- Sets up Python environment
- Configures environment variables
- Supports both GUI and CLI modes
- Note: File has no extension to avoid `.sh` gitignore pattern

## Testing

### Verified Working
- ✅ All Python package imports
- ✅ PyTorch 1.13.1 installation and import
- ✅ PyQt5 5.15.11 installation and import
- ✅ sounddevice and soundfile imports
- ✅ Toolbox, Encoder, Synthesizer, and Vocoder module imports
- ✅ CLI help command execution
- ✅ GUI toolbox startup
- ✅ Model downloads (including large files)

### Installation Methods
1. **Manual Installation**: Follow the steps in this guide
2. **Helper Script**: Run `./run_macos` (recommended)
3. **Standard uv workflow**: `uv sync --extra cpu` (after initial setup)

## Compatibility

### Supported macOS Versions
- macOS 12.3+ (for MPS support, though this project uses CPU)
- Apple Silicon (M1, M2, M3, etc.) - ARM64 ✅ Tested
- Intel Macs - x86_64 (should work but not tested)

### Python Version
- Python 3.10.x (required for PyQt5 ARM64 wheels)

## Files Changed

The following files were modified or added to support macOS:

- `pyproject.toml` - Configuration updates for macOS support
- `README.md` - Added macOS installation section
- `MACOS_INSTALL.md` - This comprehensive guide
- `run_macos` - Helper script (no extension to avoid gitignore)
- `utils/default_models.py` - Improved Google Drive download handling for large files

## Notes for Maintainers

1. The PyTorch version difference between platforms is intentional and necessary
2. Python 3.10 requirement only applies to macOS (Linux/Windows can still use 3.9)
3. The `DYLD_LIBRARY_PATH` environment variable is required at runtime
4. Consider adding a `.env` file or activation script for easier setup

## User Impact

- ✅ macOS users can now install and run the tool without manual workarounds
- ✅ Clear documentation reduces support requests
- ✅ Helper script automates the setup process
- ✅ Maintains backward compatibility with Linux/Windows

## Future Improvements

1. **CI/CD**: Add macOS runner to GitHub Actions
2. **Automation**: Improve `uv sync` to handle macOS automatically
3. **Documentation**: Add macOS-specific troubleshooting to main README
4. **Testing**: Add automated tests for macOS installation

## Contributing

If you encounter issues or have improvements for macOS support, please:
1. Test your changes thoroughly
2. Update this document if needed
3. Submit a pull request with clear description of changes

