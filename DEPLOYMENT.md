# MedGPT Cloud Run Deployment Guide

This guide walks you through deploying the Medical Records Summary Tool to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud account with billing enabled
2. **Google Cloud SDK**: Install the `gcloud` CLI tool
3. **Docker**: Install Docker on your local machine
4. **OpenAI API Key**: You'll need an OpenAI API key for the AI processing

## Quick Deployment

### 1. Set up Google Cloud Project

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Authenticate with Google Cloud
gcloud auth login
gcloud config set project $GOOGLE_CLOUD_PROJECT
```

### 2. Run the Deployment Script

```bash
# Make the script executable (if not already)
chmod +x deploy.sh

# Run the deployment
./deploy.sh
```

### 3. Set Environment Variables

After deployment, set your OpenAI API key:

```bash
gcloud run services update medgpt \
  --region=us-central1 \
  --set-env-vars OPENAI_API_KEY=your-openai-api-key-here
```

### 4. Get Your Service URL

```bash
gcloud run services describe medgpt \
  --region=us-central1 \
  --format='value(status.url)'
```

## Manual Deployment Steps

If you prefer to deploy manually:

### 1. Enable APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. Build and Deploy

```bash
# Submit build to Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Or build locally and push
docker build -t gcr.io/$GOOGLE_CLOUD_PROJECT/medgpt .
docker push gcr.io/$GOOGLE_CLOUD_PROJECT/medgpt

# Deploy to Cloud Run
gcloud run deploy medgpt \
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/medgpt \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `PORT`: Port number (automatically set by Cloud Run)
- `PYTHONUNBUFFERED`: Set to 1 for proper logging

### Resource Limits

The application is configured with:
- **Memory**: 2GB (optimized for direct PDF processing)
- **CPU**: 1 vCPU (efficient multimodal processing)
- **Timeout**: 60 minutes (for large document processing)
- **Max Instances**: 10 (auto-scaling)

## Security Considerations

1. **API Key**: Store your OpenAI API key securely using Cloud Run environment variables
2. **Authentication**: The service is deployed with `--allow-unauthenticated` for ease of use
3. **File Uploads**: Large file uploads (up to 50GB) are supported
4. **Temporary Files**: All uploaded files are automatically cleaned up

## Monitoring and Logging

### View Logs

```bash
# View recent logs
gcloud run services logs read medgpt --region=us-central1

# Follow logs in real-time
gcloud run services logs tail medgpt --region=us-central1
```

### Monitoring

- Cloud Run provides built-in monitoring for requests, latency, and errors
- Visit the Cloud Run console to view metrics and set up alerts

## Troubleshooting

### Common Issues

1. **Build Failures**: Check that all dependencies are listed in `requirements.txt`
2. **Memory Issues**: Increase memory limit if processing large files
3. **Timeout Issues**: Increase timeout for very large document sets
4. **API Key Errors**: Ensure OPENAI_API_KEY is set correctly

### Debug Commands

```bash
# Check service status
gcloud run services describe medgpt --region=us-central1

# View environment variables
gcloud run services describe medgpt --region=us-central1 --format="value(spec.template.spec.template.spec.containers[0].env[].name,spec.template.spec.template.spec.containers[0].env[].value)"

# Update environment variables
gcloud run services update medgpt --region=us-central1 --set-env-vars KEY=value
```

## Cost Optimization

- **Auto-scaling**: Cloud Run scales to zero when not in use
- **Resource Limits**: Configured for optimal performance vs. cost
- **Regional Deployment**: Using us-central1 for cost efficiency

## Updating the Application

To update the deployed application:

```bash
# Build and deploy new version
gcloud builds submit --config cloudbuild.yaml

# Or using the deployment script
./deploy.sh
```

## Cleanup

To remove the deployment:

```bash
# Delete the Cloud Run service
gcloud run services delete medgpt --region=us-central1

# Delete container images (optional)
gcloud container images delete gcr.io/$GOOGLE_CLOUD_PROJECT/medgpt
```

## Support

For issues with this deployment:
1. Check the Cloud Run logs for errors
2. Verify all environment variables are set correctly
3. Ensure your Google Cloud project has sufficient quotas
4. Contact support if you encounter persistent issues