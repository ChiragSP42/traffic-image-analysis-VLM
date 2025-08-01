from typing import List, Optional, Any

def list_obj_s3(s3_client: Any,
                bucket_name: Optional[str],
                folder_name: Optional[str],
                delimiter: Optional[str] = '')-> List[str]:
    """
    Function to return list of objects present in bucket. There is an optional
    delimiter parameter to toggle between folder and file names. If delimiter is empty, it will return all files in the bucket.

    Parameters:
        s3_client (Any): S3 client object
        bucket_name (str): Name of S3 bucket where concerned docs are present.
        foldername (str): Name of folder in which pdfs are present.
        delimiter (str): Delimiter to toggle between folder and file names. Default is '/'.

    Returns:
        pdf_list (list[str]): List of pdf names with folder path included.
    """

    pdf_list = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name,
                                   Prefix=folder_name,
                                   Delimiter=delimiter):
        if delimiter:
            if 'CommonPrefixes' in page:
                pdf_list = [obj["Prefix"] for obj in page.get('CommonPrefixes', [])]
        else:
            for obj in page.get('Contents', []):
                key = obj['Key']
                pdf_list.append(key)

    return pdf_list