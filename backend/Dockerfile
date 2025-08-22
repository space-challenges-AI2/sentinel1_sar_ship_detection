# 1. Use an official, lightweight Python base image
FROM python:3.9-slim

# 1.2 Install the missing system dependency for OpenCV
RUN apt-get update && apt-get install -y libgl1

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy and install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code into the container
# This includes main.py, the yolov5 folder, and your best.pt file
COPY . .

# 5. Tell Docker the app will run on port 8080 (standard for Cloud Run)
EXPOSE 8080

# 6. The command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
