from time import sleep
from prefect_aws import S3Bucket, AWSCredentials

def create_aws_credentials_block():
    my_aws_creds_obj = AWSCredentials(
        aws_acces_key_id="123abc", aws_secret_access_key="abs123"
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite = True)

def create_s3_bucket_block():
    aws_creds = AWSCredentials.load('my-aws-creds')
    my_s3_bucket_obj = S3Bucket(
        bucket_name = "mri-predict", credentials =  aws_creds
    )
    my_s3_bucket_obj.save(name="mir-predict", overwrite = True)

if __name__='__main__':
    create_aws_credentials_block()
    sleep(5)
    create_s3_bucket_block()