FROM python:3.12-slim

# Shared libraries OpenCV links against.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY configs ./configs
COPY src ./src

# CPU PyTorch keeps the image small. Ultralytics pulls in opencv-python and EasyOCR
# pulls in the headless build. They overwrite each other, so keep only the headless one.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[ml]" \
    && pip uninstall --yes opencv-python opencv-python-headless \
    && pip install --no-cache-dir --no-deps opencv-python-headless

RUN useradd --create-home vrpr && mkdir -p /data/output && chown -R vrpr /data
USER vrpr

# Mount weights and videos under /data, and collect results from /data/output.
ENTRYPOINT ["vrpr"]
CMD ["--help"]
