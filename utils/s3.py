import os
import boto3
from dotenv import dotenv_values

BUCKET_NAME = "poc-izanagi"
config = dotenv_values()

class S3:
    def __init__(self, bucket_name=BUCKET_NAME, region="us-east-2"):
        self.bucket = bucket_name
        self.s3 = boto3.resource(
            service_name="s3",
            region_name=region,
            aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"],
        )

    def download_data(self, local_dir, prefix=""):
        """
        Baixa todo o conteúdo do bucket (ou de um sub-diretório 'prefix')
        para 'local_dir', preservando a estrutura de pastas.
        """
        bucket = self.s3.Bucket(self.bucket)
        for obj in bucket.objects.filter(Prefix=prefix):

            # ignora "pastas" vazias 
            if obj.key.endswith("/"):
                continue

            local_path = os.path.join(local_dir, obj.key)
            local_folder = os.path.dirname(local_path)
            os.makedirs(local_folder, exist_ok=True)

            print(f"↓ Baixando s3://{self.bucket}/{obj.key} → {local_path}")
            bucket.download_file(obj.key, local_path)

if __name__ == "__main__":
    downloader = S3()
    downloader.download_data(local_dir="./data_s3")

