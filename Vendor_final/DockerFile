# Use official Python base image
FROM python:3.9

# Create app directories
RUN mkdir -p /app/src
RUN mkdir -p /app/model
RUN mkdir -p /app/data

# Set working directory
WORKDIR /app/src

# Copy files into container
COPY app.py /app/src/
COPY requirements.txt /app/src/
COPY VENDOR_FINAL.xlsx /app/data/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set default command
CMD ["python", "app.py"]
