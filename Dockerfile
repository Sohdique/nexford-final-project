FROM python:3.13.7

# set a directory for the app
WORKDIR /app

# copy requirements first for better caching
COPY requirements.txt .

# install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy all the files to the container
COPY . .

# create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# define the port number the container should expose
EXPOSE 5000

# run the command app or application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]