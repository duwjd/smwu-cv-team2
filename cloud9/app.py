import streamlit as st
import boto3
import json
import numpy as np
from PIL import Image
import io
import pymysql # mysql에 분석 결과 저장
import base64
import nvgpu

# AWS 설정
def get_predictor():
    runtime = boto3.client(
        'runtime.sagemaker',
        region_name='us-east-1',  # 예: 'ap-northeast-2'
        aws_access_key_id='...',
        aws_secret_access_key='...'
    )
    return runtime

# sagemaker로 이미지 예측 요청
def predict_image(image_bytes, endpoint_name):
    runtime = boto3.client('sagemaker-runtime')  # SageMaker Runtime 클라이언트 생성

    # 이미지를 base64로 인코딩하고 JSON 형식으로 감싸기
    img_json = {
        "image": base64.b64encode(image_bytes).decode('utf-8')  # 바이트 배열을 base64 문자열로 변환
    }

    # SageMaker 엔드포인트 호출
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',  # JSON 형식으로 Content-Type 설정
        Body=json.dumps(img_json)        # JSON 문자열로 변환하여 전송
    )

    # 결과 반환
    result = json.loads(response['Body'].read().decode())  # 응답 JSON 디코딩
    return result

# EC2 MySQL 데이터베이스 연결 함수
def get_db_connection():
    connection = pymysql.connect(
        host="54.224.14.161",       # EC2 퍼블릭 IP
        user="root",             # MySQL 사용자 이름
        password="",     # MySQL 비밀번호
        database="image_classification"  # 데이터베이스 이름
    )
    return connection

# 분류 결과 DB에 저장
def save_result_to_db(image_name, classification_result):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO results (image_name, classification_result) VALUES (%s, %s)"     # 테이블명 : results
            cursor.execute(sql, (image_name, classification_result))
            connection.commit()
    finally:
        connection.close()


def main():
    st.title('다이캐스팅 품질 판별 시스템')
    
    uploaded_file = st.file_uploader("불량 여부를 판단하려는 이미지를 업로드해주세요", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption='업로드된 이미지', use_container_width=True)
        
        # 예측 버튼
        if st.button('분류 시작'):
            with st.spinner('분류 중...'):
                # 이미지를 바이트로 변환
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()
                
                try:
                    # SageMaker 엔드포인트 호출 (JSON 형식으로 전송)
                    result = predict_image(
                        img_byte_arr,          # 바이트 배열을 predict_image로 전달
                        '34-endpoint'    # SageMaker 엔드포인트 이름
                    )
                    
                    # 결과 처리
                    classification_result = result.get('prediction', '예측 불가')  # SageMaker 결과에서 예측 라벨 추출

                    # 결과를 DB에 저장
                    save_result_to_db(uploaded_file.name, classification_result)
                    
                    # 결과 표시
                    st.success('분류가 완료되었습니다!')
                    st.write(f'분류 결과: {classification_result}')
                    
                except Exception as e:
                    st.error(f'오류가 발생했습니다: {str(e)}')


    
if __name__ == '__main__':
    main()
    