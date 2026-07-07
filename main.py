import os
import sys
import pickle
import traceback
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# 1. 상위 폴더(c:\Users\user\앤티그래비티)를 sys.path에 추가하여 hybrid_model 임포트 가능케 함
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from hybrid_model import HybridFireDetector
except ImportError:
    # 혹시 상위 폴더에 없거나 복사가 필요한 경우를 대비해 폴백 정의
    class HybridFireDetector:
        def __init__(self, isolation_forest_model=None, temp_threshold=55.0, co_threshold=80.0, feature_names=None):
            self.temp_threshold = temp_threshold
            self.co_threshold = co_threshold
        def predict(self, X):
            # scikit-learn 모델이 없을 경우의 룰베이스 폴백 예측
            if isinstance(X, pd.DataFrame):
                temp = X.iloc[0, 0]
                co = X.iloc[0, 2]
            else:
                temp = X[0][0]
                co = X[0][2]
            # 룰 기반 탐지 (온도 > 55.0 또는 CO > 80.0 이면 화재/-1)
            if temp > self.temp_threshold or co > self.co_threshold:
                return np.array([-1])
            return np.array([1])

# 2. 환경변수 및 Supabase 초기화
load_dotenv(os.path.join(current_dir, ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase_client = None
if SUPABASE_URL and SUPABASE_KEY and "your_supabase" not in SUPABASE_URL:
    try:
        from supabase import create_client, Client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[SUCCESS] Supabase Client 초기화 완료.")
    except Exception as e:
        print(f"[WARNING] Supabase Client 초기화 실패 (라이브러리 미설치 혹은 설정 에러): {e}")
else:
    print("[WARNING] Supabase 설정이 비어있거나 기본 플레이스홀더입니다. Supabase 저장 기능은 비활성화(로그로 대체)됩니다.")

# 3. 모델 로드 함수 정의
MODEL_PATH = os.path.join(current_dir, "hybrid_anomaly_model.pkl")
model = None

def load_active_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print(f"[SUCCESS] {MODEL_PATH} 로드 완료.")
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            traceback.print_exc()
            model = None
    else:
        print(f"[WARNING] {MODEL_PATH} 파일이 없습니다. 임시 폴백 모델을 작동시킵니다.")
        model = None

# 서버 시작 시 모델 로드
load_active_model()

# 4. FastAPI 앱 설정
app = FastAPI(title="PWA Backend for Fire Detection", description="Fire anomaly detection server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 스키마 정의
class SensorPayload(BaseModel):
    temperature: float
    humidity: float
    co: float

@app.get("/status")
def get_status():
    return {
        "model_loaded": model is not None,
        "supabase_connected": supabase_client is not None,
        "model_path": MODEL_PATH,
        "supabase_url": SUPABASE_URL
    }

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Only .pkl files are allowed.")
    
    try:
        content = await file.read()
        with open(MODEL_PATH, "wb") as f:
            f.write(content)
        
        # 모델 다시 로드
        load_active_model()
        
        return {"status": "success", "message": "Model uploaded and loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

@app.post("/detect")
async def detect_anomaly(payload: SensorPayload):
    # 피처명 매핑 처리: 딕셔너리 모델인 경우 features를 따름. 기본값은 영문 및 한글 양쪽 폴백 처리
    feature_names = ['Temperature_C', 'Humidity_Percent', 'CO_ppm']
    if model is not None and isinstance(model, dict):
        feature_names = model.get('features', feature_names)
        
    input_df = pd.DataFrame([[
        payload.temperature,
        payload.humidity,
        payload.co
    ]], columns=feature_names)
    
    # 1. 이상 탐지 수행 (1: 정상, -1: 이상/화재)
    prediction_val = 1
    
    if model is not None:
        try:
            if isinstance(model, dict):
                # 딕셔너리 모델 구조 (scaler, model, features) 대응
                scaler = model.get('scaler')
                clf = model.get('model')
                
                # 스케일러로 전처리 후 Isolation Forest 예측
                if scaler is not None:
                    scaled_data = scaler.transform(input_df)
                else:
                    scaled_data = input_df
                    
                pred = clf.predict(scaled_data)
                prediction_val = int(pred[0])
            else:
                # HybridFireDetector 인스턴스인 경우 한글 컬럼명 사용
                input_df_kr = pd.DataFrame([[
                    payload.temperature,
                    payload.humidity,
                    payload.co
                ]], columns=['온도', '습도', '일산화농도'])
                pred = model.predict(input_df_kr)
                prediction_val = int(pred[0])
        except Exception as e:
            print(f"[ERROR] Model Prediction Error: {e}")
            # 폴백 예측 작동 (룰베이스 적용)
            fallback_model = HybridFireDetector()
            # fallback_model은 한글 컬럼명을 기본으로 받으므로 kr 포맷 사용
            input_df_kr = pd.DataFrame([[
                payload.temperature,
                payload.humidity,
                payload.co
            ]], columns=['온도', '습도', '일산화농도'])
            pred = fallback_model.predict(input_df_kr)
            prediction_val = int(pred[0])
    else:
        # 폴백 예측 작동 (룰베이스 적용)
        fallback_model = HybridFireDetector()
        input_df_kr = pd.DataFrame([[
            payload.temperature,
            payload.humidity,
            payload.co
        ]], columns=['온도', '습도', '일산화농도'])
        pred = fallback_model.predict(input_df_kr)
        prediction_val = int(pred[0])
        
    # 절대 안전 임계치 초과 시 룰베이스 강제 화재 판정 (-1)
    # 온도 > 55.0 도는 CO > 80.0ppm 인 경우
    if payload.temperature > 55.0 or payload.co > 80.0:
        prediction_val = -1
        
    # 결과값(RESULT) 판정 및 맵핑 규칙 적용
    # 2: 비정상 (prediction_val == -1)
    # 1: 정상이며 온도가 40도 이상
    # 0: 정상이며 온도가 40도 미만
    if prediction_val == -1:
        result_val = 2
    else:
        if payload.temperature >= 40.0:
            result_val = 1
        else:
            result_val = 0
            
    # 2. Supabase 저장 (MT01 테이블)
    supabase_success = False
    supabase_error = None
    if supabase_client is not None:
        try:
            data = {
                "TEMP": payload.temperature,
                "HUMI": payload.humidity,
                "CO": payload.co,
                "RESULT": result_val
            }
            res = supabase_client.table("MT01").insert(data).execute()
            supabase_success = True
        except Exception as e:
            supabase_error = str(e)
            print(f"[ERROR] Supabase Insert Error: {e}")
    else:
        print(f"[INFO] Supabase 미연동 상태로 DB 저장 생략. 데이터: TEMP={payload.temperature}, HUMI={payload.humidity}, CO={payload.co}, RESULT={result_val}")
    
    # 3. 결과 반환 (PC 클라이언트로 송신)
    return {
        "temperature": payload.temperature,
        "humidity": payload.humidity,
        "co": payload.co,
        "result": result_val
    }

# 5. 정적 파일 마운트 (PWA 프론트엔드 서빙)
# API 경로가 덮어씌워지지 않도록 앱 하단에 마운트합니다.
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory=current_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
