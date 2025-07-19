FROM python:3.11-slim

# Install cron
RUN apt-get update && apt-get install -y cron

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Setup cron
COPY cronjob /etc/cron.d/cronjob
RUN chmod 0644 /etc/cron.d/cronjob
RUN crontab /etc/cron.d/cronjob

# Create log file
RUN touch /var/log/cron.log

EXPOSE 8000

# Start cron and app
CMD bash -c "cron && uvicorn app.main:app --host 0.0.0.0 --port 8000"