# Use python 3.11 as base image
FROM python:3.11-slim

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install poetry 
RUN pip install poetry

# copy only dependency files first
#COPY pyproject.toml poetry.lock README.md ./
COPY . .

# install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# copy the rest of the application
COPY . .


# create directories for uploads and ensure proper permissions
RUN mkdir -p web/static/uploads web/static/output \
    && chmod -R 777 web/static/uploads web/static/output


# Expose port
EXPOSE 8000

# RUN the application
CMD ["poetry", "run", "uvicorn", "web.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
