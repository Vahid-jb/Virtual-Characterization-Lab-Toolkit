#!/bin/bash

# Exit immediately if any command fails
set -e

echo "🧹 Cleaning up old build artifacts..."
sudo rm -rf build/ dist/ vcl_win_x64/

echo "🚀 Starting Windows compilation via Docker..."

# We use a Heredoc (<< 'EOF') to feed the script to Docker cleanly.
# Notice we use -i instead of -it here, which is required when passing scripts this way.
sudo docker run -i --rm \
    -v "$(pwd):/src/" \
    -v pyinstaller_pip_cache:/pip_cache \
    pyinstaller-win-3.12 /bin/bash << 'EOF'

echo "🖥️  Starting virtual display..."
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99
sleep 1

echo "📦 Installing Python requirements (using cache)..."
export PIP_CACHE_DIR="Z:\pip_cache"
wine /wine/drive_c/Python312/python.exe -m pip install -r requirements.txt

echo "🔨 Running PyInstaller..."
wine /wine/drive_c/Python312/Scripts/pyinstaller.exe --noconfirm merged.spec

echo "🛑 Shutting down Wine background processes..."
wineserver -k

EOF
# ^^^ The EOF above must be exactly at the start of the line with no spaces before or after it!

echo "✅ Build complete! Taking ownership of output files..."
# Change ownership from root back to your current Linux user
sudo chown -R $USER:$USER dist/ build/

echo "📦 Packaging the release..."
BUILD_DIR=$(find dist/ -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [ -z "$BUILD_DIR" ]; then
    echo "❌ Could not find a folder inside dist/. Did PyInstaller build a single .exe instead?"
    exit 1
fi

echo "Copying $BUILD_DIR to vcl_win_x64..."
cp -r "$BUILD_DIR" vcl_win_x64

echo "🗜️ Compressing to zip..."
zip -rq vcl_win_x64.zip vcl_win_x64/

echo "📁 Moving to releases folder..."
mkdir -p releases
mv vcl_win_x64.zip releases/

# Clean up the temporary staging folder
rm -rf vcl_win_x64

echo "🎉 All done! Your packaged Windows build is ready at releases/vcl_win_x64.zip"