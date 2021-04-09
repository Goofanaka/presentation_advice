from __future__ import print_function
import time
import boto3
import re
from urllib import request
import json
from ast import literal_eval

pattern = re.compile(r'.+')

# 음성 파일 경로를 넣어주세요 by.기훈
file_name = 'JUA_SOUND.wav'
# 음성 파일을 저장할 아마존 S3 버켓입니다. by.기훈
bucket_name = 'goofanaka-stt-test'

# boto3는 아마존 aws 서비스를 파이썬에서 사용하기 위한 라이브러리 입니다.
# 없다면 꼭 설치를 바랍니다. by.기훈
s3 = boto3.client('s3', # 사용할 서비스 이름 by.기훈
                  aws_access_key_id='AKIA547LEYTTLKBJUJXQ', # 액세스키 by.기훈
                  aws_secret_access_key='CcRQnFjzVYdH6NGemALtcTrnLs46c/nRCDSsmO7v', # 시크릿 키 by.기훈
                  region_name='ap-northeast-2' # 사용하는 서버 위치 by.기훈
                  )

# s3 버켓에 업로드하는 코드 입니다. by.기훈
s3.upload_file(file_name , # 버켓에 저장할 파일의 경로 입니다.
               bucket_name, # 저장 할 버켓의 이름입니다.
               file_name) # 버켓에 저장될 파일 이름입니다.

## 여기서 부터는 aws transcribe에 넣기 위한 코드입니다.

# s3 버켓의 경로입니다.
s3_uri = f's3://{bucket_name}/'
print(s3_uri)
file_format = file_name[file_name.find('.')+1:]
print(file_format)
transcribe = boto3.client('transcribe',
                          aws_access_key_id='AKIA547LEYTTLKBJUJXQ',
                          aws_secret_access_key='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',
                          region_name='ap-northeast-2'
                          )
job_name = 'KDG-TEST'
job_uri = f"s3://{bucket_name}/{file_name}"
transcribe.start_transcription_job(
    TranscriptionJobName=job_name, # 중복 노노, 이름은 자유롭게 설정 가능
    Media={'MediaFileUri': job_uri}, # s3 버켓에 올라가 있는 음성 파일 경로(s3-uri)
    MediaFormat=file_format, # 파일의 포맷(m4a = mp4, mp3, wav, flac...)
    LanguageCode='ko-KR' # 언어 설정
)

while True:
    status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    print("Not ready yet...")
    time.sleep(5)
print(status)

# json 다운로드
'''
url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
savename = 'test.json'
request.urlretrieve(url, savename)
'''
#json 변수에 할당
'''
url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
savename = 'test.json'
text = request.urlopen(url).read()
text = literal_eval(text.decode('utf-8'))
s = json.dumps(text, indent=4, sort_keys=True)
print(text)
print(s)
'''