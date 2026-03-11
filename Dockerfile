FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python deps for analysis
RUN pip3 install --no-cache-dir numpy matplotlib

# Copy source
WORKDIR /app
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY python/ python/
COPY scripts/ scripts/

# Build
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" && \
    make -j$(nproc)

# Create output directories
RUN mkdir -p data figures

# Default: run simulation with 100M photons
ENTRYPOINT ["/app/build/mc_fnirs"]
CMD ["--photons", "100000000", "--output", "/app/data"]
