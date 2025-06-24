# Configuration Guide

## Azure Storage Setup

To use real Azure Storage instead of mock mode, you need to configure a valid Azure Storage connection string.

### Option 1: Environment Variable (Recommended)

Set the `AZURE_CONNECTION_STRING` environment variable:

**Windows PowerShell:**
```powershell
$env:AZURE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net"
```

**Windows Command Prompt:**
```cmd
set AZURE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net
```

**Linux/macOS:**
```bash
export AZURE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net"
```

### Option 2: .env File

Create a `.env` file in the project root with:
```
AZURE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net
```

### Getting Your Azure Connection String

1. Go to the Azure Portal
2. Navigate to your Storage Account
3. Go to "Access keys" under "Security + networking"
4. Copy the connection string from either key1 or key2

### Current Status

The application is currently using **mock mode** because:
- No valid Azure connection string is configured
- The default test connection string is being used

When mock mode is active, the application will:
- Create mock files instead of downloading from Azure
- Generate realistic mock data for testing
- Work without requiring Azure Storage access

### Troubleshooting

If you see the error: `Failed to resolve 'defaultendpointsprotocol=https'`, it means:
1. The connection string format is incorrect
2. You're using the test connection string instead of a real one
3. The application will automatically fall back to mock mode

To fix this, provide a valid Azure Storage connection string using one of the methods above. 