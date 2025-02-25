#!/bin/bash

poetry run uvicorn web.api.main:app --reload --host 0.0.0.0 --port 8000