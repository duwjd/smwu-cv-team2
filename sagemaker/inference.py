import torch
from torchvision import models, transforms
from PIL import Image
import json
import base64
from io import BytesIO

def model_fn(model_dir):
    # 모델 파일 경로 설정
    model_path = f"{model_dir}/best_model_34.pth"
    print(f"Loading model from: {model_path}")

    try:
        # GPU가 있는지 확인
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Model will run on GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("No GPU available. Model will run on CPU.")

        model = models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 클래스 개수: 2개

        # 학습된 가중치 로드 (디바이스에 맞게 로드)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)  # 모델을 디바이스로 이동
        model.eval()  # 평가 모드로 설정
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def input_fn(request_body, request_content_type):
    """
    SageMaker가 호출하는 입력 데이터 처리 함수
    """
    if request_content_type == "application/json":
        # JSON 데이터에서 이미지 경로 추출
        input_data = json.loads(request_body)
        #인코딩된 이미지데이터 디코딩
        image_data = base64.b64decode(input_data["image"])
        #디코딩된 데이터 PIllow이미지로 변환
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    SageMaker가 호출하는 추론 함수
    """
    # 데이터 전처리
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 입력 크기로 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 전처리
    image_tensor = preprocess(input_data).unsqueeze(0)

    # 모델 추론
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 클래스 반환
    return predicted.item()

def output_fn(prediction, response_content_type):
    """
    SageMaker가 호출하는 출력 데이터 포맷팅 함수
    """
    if response_content_type == "application/json":
        
        result = "OK" if prediction == 1 else "NG"
        
        return json.dumps({"result": result})  # JSON 형식으로 결과 반환
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
