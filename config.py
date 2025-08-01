import os
from dotenv import load_dotenv

environment = os.getenv('ENVIRONMENT', 'development')
if environment == 'production':
    load_dotenv('.env.production')
else:
    load_dotenv()

BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:8000')
EXTERNAL_API_BASE_URL = os.getenv('EXTERNAL_API_BASE_URL', 'https://localhost:44302')
EXTERNAL_API_KEY = os.getenv('EXTERNAL_API_KEY', 'WID-API-2024-SecureKey-XYZ789')
AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING', 'DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net')
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')
ENVIRONMENT = environment 