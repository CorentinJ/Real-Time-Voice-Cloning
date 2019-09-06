import logging
from pathlib import Path

from google.cloud import storage

from encoder.inference import Model


def download_blob(bucket_name, source_blob_name, destination_file_name, gcloud_project):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client(project=gcloud_project)
    # don't use get_bucket which requires storage.buckets.get permission
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    logging.info('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name)
    )


def load_model(gcs_bucket_name: str, gcs_blob_name: str, gcloud_project: str) -> Model:
    model_path = Path('/tmp/encoder_pretrained.pt')
    if not model_path.is_file():
        download_blob(gcs_bucket_name, gcs_blob_name, model_path, gcloud_project)

    model = Model()
    model.load(Path(model_path))
    return model
