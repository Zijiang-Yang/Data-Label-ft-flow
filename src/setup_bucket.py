import os
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

os.environ['OSS_ACCESS_KEY_ID'] = 'xxxx'
os.environ['OSS_ACCESS_KEY_SECRET'] = 'xxxx'

auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
bucket = oss2.Bucket(auth, 'https://oss-cn-shenzhen.aliyuncs.com', 'xxxx')


def bucket_viewer(project_name):
    dir_path = os.path.join("data_label_ft_flow", project_name)
    for obj in oss2.ObjectIteratorV2(bucket, prefix=dir_path):
        print(obj.key)


if __name__ == "__main__":
    bucket_viewer("")