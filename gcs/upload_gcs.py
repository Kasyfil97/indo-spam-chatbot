from google.cloud import storage
import os

def upload_directory_to_gcs(source_directory_path, destination_blob_prefix, bucket_name = 'llm-buckets'):
    """Uploads a directory to a GCS bucket.

    Args:
        bucket_name: The name of the GCS bucket.
        source_directory_path: The path to the directory to upload, relative to the current working directory.
        destination_blob_prefix: The prefix to add to the blobs on GCS (can be a folder structure).
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for local_root, _, files in os.walk(source_directory_path):
        for file in files:
            local_path = os.path.join(local_root, file)
            # Construct the blob's name in the bucket.
            relative_path = os.path.relpath(local_path, source_directory_path)
            blob_name = os.path.join(destination_blob_prefix, relative_path).replace("\\", "/")  # Ensure proper path format in GCS.
            
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to {blob_name}.")