#!/usr/bin/env python3
"""
Test script for B2 upload functionality
"""
import os
import sys
import time
from pathlib import Path
import boto3

# B2 Storage configuration
B2_CONFIG = {
    'access_key_id': '005535c6992951a0000000001',
    'secret_access_key': 'K005pcqp/ctlk4HBOGZF7lWWDL4Qg3k',
    'region': 'us-east-005',
    'endpoint_url': 'https://s3.us-east-005.backblazeb2.com',
    'bucket_name': 'ahmadhannanmassod'
}

def upload_to_b2(file_path):
    """
    Upload a file to Backblaze B2 storage and return the URL
    """
    try:
        # Create S3 client with B2 configuration
        s3_client = boto3.client(
            's3',
            region_name=B2_CONFIG['region'],
            endpoint_url=B2_CONFIG['endpoint_url'],
            aws_access_key_id=B2_CONFIG['access_key_id'],
            aws_secret_access_key=B2_CONFIG['secret_access_key']
        )
        
        # Get the filename from the path
        file_name = os.path.basename(file_path)
        
        # Add timestamp to ensure uniqueness
        timestamp = int(time.time())
        unique_file_name = f"{timestamp}_{file_name}"
        
        print(f"Uploading {file_path} to B2 as {unique_file_name}...")
        
        # Upload the file
        s3_client.upload_file(
            file_path,
            B2_CONFIG['bucket_name'],
            unique_file_name
        )
        
        # Generate the URL
        url = f"{B2_CONFIG['endpoint_url']}/{B2_CONFIG['bucket_name']}/{unique_file_name}"
        print(f"File uploaded successfully. URL: {url}")
        
        return url
    except Exception as e:
        print(f"Error uploading to B2: {str(e)}")
        return None

def create_test_file(path, size_kb=10):
    """Create a test file of specified size in KB"""
    with open(path, 'wb') as f:
        f.write(b'0' * (size_kb * 1024))
    return path

def main():
    # Create a test file
    test_file_path = "/tmp/test_upload.bin"
    create_test_file(test_file_path)
    
    # Test the upload function
    url = upload_to_b2(test_file_path)
    
    if url:
        print(f"✅ Test passed! File uploaded successfully to: {url}")
        return 0
    else:
        print("❌ Test failed! Could not upload file to B2.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
