# Use Ubuntu 24.04 as our base
FROM ubuntu:24.04

# Suppress interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive

# Enable 32-bit architecture and install tools including unzip
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    wget ca-certificates xvfb gnupg unzip zip && \
    rm -rf /var/lib/apt/lists/*

# Add WineHQ repo (using HTTP to bypass TLS bugs) and install Wine 10.0
RUN mkdir -pm755 /etc/apt/keyrings && \
    wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key && \
    wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources && \
    sed -i 's/https/http/g' /etc/apt/sources.list.d/winehq-noble.sources && \
    apt-get update && \
    apt-get install -y --install-recommends winehq-stable && \
    rm -rf /var/lib/apt/lists/*

# Configure Wine environment variables
ENV WINEPREFIX=/wine \
    WINEARCH=win64 \
    WINEDEBUG=-all \
    WINEDLLOVERRIDES="mscoree,mshtml="

# Initialize Wine, set to Windows 10
RUN xvfb-run -a sh -c 'wineboot --init && winecfg -v win10 && wineserver -w'

# Install Windows Python 3.12
RUN wget https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe -O python-installer.exe && \
    xvfb-run -a sh -c 'wine python-installer.exe /quiet InstallAllUsers=1 TargetDir=C:\\Python312 PrependPath=1 Include_test=0 && wineserver -w' && \
    rm python-installer.exe

# Install Microsoft Visual C++ Redistributable (Required by scientific packages)
RUN wget https://aka.ms/vs/17/release/vc_redist.x64.exe -O vc_redist.x64.exe && \
    xvfb-run -a sh -c 'wine vc_redist.x64.exe /quiet /norestart && wineserver -w' && \
    rm vc_redist.x64.exe

# Inject ICU Unicode DLLs directly into Wine (Required by Qt6 / PySide6)
RUN wget https://github.com/unicode-org/icu/releases/download/release-74-2/icu4c-74_2-Win64-MSVC2019.zip -O icu.zip && \
    unzip icu.zip -d icu_tmp && \
    cp icu_tmp/bin64/icu*.dll /wine/drive_c/windows/system32/ && \
    cd /wine/drive_c/windows/system32/ && \
    cp icuuc74.dll icuuc.dll && cp icuin74.dll icuin.dll && cp icudt74.dll icudt.dll && \
    rm -rf /icu.zip /icu_tmp

# Install PyInstaller
RUN xvfb-run -a wine /wine/drive_c/Python312/python.exe -m pip install pyinstaller

# Set the working directory for when the container runs
WORKDIR /src