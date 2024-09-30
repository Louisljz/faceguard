import uuid
import datetime
from google.cloud import storage


IMAGE_SIZE = (1280, 720)


class GCSUtil:

    def __init__(self, bucket_name, service_account_dict):
        self.client = storage.Client.from_service_account_info(service_account_dict)
        self.bucket = self.client.bucket(bucket_name)

    def upload_video(self, video, expiration=1):
        blob = self.bucket.blob(f"{uuid.uuid4()}.mp4")
        blob.upload_from_file(video)

        expiration = datetime.timedelta(hours=expiration)
        url = blob.generate_signed_url(expiration=expiration, method="GET")
        return url
