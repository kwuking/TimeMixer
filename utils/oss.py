"""
    Copyright (c) 2018-present, Ant Financial Service Group
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
   ------------------------------------------------------
    # @File: oss.py
    # @Create Time: 2023/3/15 7:34 下午
    # @Author: weiming.wsy
    # @Comments:
"""


def get_file_from_oss(
        oss_uri: str,
        local_path: str = '.',
        oss_endpoint: str = None
):
    """ Download file from oss.

    Args:
        oss_uri: str
            File path in oss.
        local_path: str
            Local target directory or file path that file storing. A folder or a exact file path.
        oss_endpoint: str
            Endpoint of oss.
    Returns:
        Local path: str. File path in local environment. Return None when file is not existed in oss.
    """
    import os
    import oss2
    from pypai.io.file.oss_client import OSSClient

    # Parse oss bucket and key from oss uri or path (key)
    prefix = "oss://"
    if not oss_uri.startswith(prefix):
        oss_uri = prefix + oss_uri
    tokens = oss_uri[len(prefix):].split('/')
    filename = tokens[-1]
    oss_bucket = tokens[0]
    oss_key = '/'.join(tokens[1:])
    # process endpoint
    if oss_endpoint is None:
        # set default endpoint to the personal endpoint
        oss_endpoint = "oss-cn-hangzhou-zmf.aliyuncs.com"
    elif oss_endpoint.startswith('http://'):
        oss_endpoint = oss_endpoint[7:]

    # Construct local file name
    if os.path.isdir(local_path):
        local_name = os.path.join(local_path, filename)
    else:
        local_name = local_path

    # Download file depend on bucket
    if oss_bucket in ["cmps-model", "dmsint", 'weiming2']:
        oss_client = OSSClient()
        if oss_client.exists(oss_uri):
            oss_client.get_file(oss_uri, local_name)
            return local_name
        else:
            return None
    else:
        # get access id and key
        oss_access_id = os.getenv('OSS_ACCESS_ID') or os.getenv("ENV_ODPS_ACCESS_ID")
        oss_access_key = os.getenv('OSS_ACCESS_KEY') or os.getenv("ENV_ODPS_ACCESS_KEY")

        bucket = oss2.Bucket(oss2.Auth(oss_access_id, oss_access_key), oss_endpoint, oss_bucket)
        if bucket.object_exists(oss_key):
            oss2.resumable_download(bucket, oss_key, local_name)
            return local_name
        else:
            return None


def save_file_to_oss(
        local_fname,
        oss_uri: str,
        oss_endpoint=None
):
    """ Upload a file or folder to oss. It will compress folder automatically when 'local_fname' is a folder.

    Args:
        local_fname: str
            Path of source local file or folder.
        oss_uri: str
            Target oss path.
        oss_endpoint: str
            Endpoint of oss.

    Returns:
        Oss uri: str. Full oss uri.
        Http url: str. Downloadable http url.
    """
    import os
    import oss2
    from pypai.io.file.oss_client import OSSClient

    # Check local file and process if it's a folder
    if not os.path.exists(local_fname):
        raise RuntimeError(f'File not exist: {local_fname}')

    # zip folder to one file
    if os.path.isdir(local_fname):
        local_fname = tar(local_fname)

    # Parse oss bucket and key from oss uri or path (key)
    # strip prefix if exists
    prefix = "oss://"
    if not oss_uri.startswith(prefix):
        oss_uri = prefix + oss_uri
    tokens = oss_uri[len(prefix):].split('/')
    oss_bucket = tokens[0]
    oss_key = '/'.join(tokens[1:])
    # process endpoint
    oss_endpoint = oss_endpoint or os.getenv('OSS_ENDPOINT')
    if oss_endpoint is None:
        # set default endpoint to the personal endpoint
        oss_endpoint = "oss-cn-hangzhou-zmf.aliyuncs.com"
    elif oss_endpoint.startswith('http://'):
        oss_endpoint = oss_endpoint[7:]

    # Upload file depend on bucket
    if oss_bucket in ["cmps-model", "dmsint"]:
        # common bucket use this way to upload file
        oss_endpoint = "cn-hangzhou.alipay.aliyun-inc.com"
        oss_client = OSSClient()
        oss_client.put_file(local_fname, oss_uri)
    else:
        # personal bucket use oss2 to upload file
        oss_endpoint = oss_endpoint or "oss-cn-hangzhou-zmf.aliyuncs.com"

        # get access id and key
        oss_access_id = os.getenv('OSS_ACCESS_ID') or os.getenv("ENV_ODPS_ACCESS_ID")
        oss_access_key = os.getenv('OSS_ACCESS_KEY') or os.getenv("ENV_ODPS_ACCESS_KEY")

        bucket = oss2.Bucket(oss2.Auth(oss_access_id, oss_access_key), oss_endpoint, oss_bucket)
        oss2.resumable_upload(bucket, oss_key, local_fname)

    http_url = 'http://%s.%s/%s' % (oss_bucket, oss_endpoint, oss_key)

    return oss_uri, http_url

