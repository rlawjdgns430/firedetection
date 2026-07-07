import numpy as np
import pandas as pd
from datetime import timedelta

def generate_sensor_data(output_path='synthetic_sensor_data.csv', seed=42):
    """
    C:\\Users\\user\\Desktop\\화재진단데이터\\전처리\\클리닝\\sensor_data.csv의 통계적 특성을 반영하여,
    정상 상태와 화재 징후(이상치)를 포함한 1만 건의 가상 데이터를 생성하는 스크립트입니다.
    """
    np.random.seed(seed)
    n_records = 10000
    
    # 1. 타임스탬프 생성 (5초 간격으로 10,000건 생성)
    # 원본 데이터가 2026-07-07 10:19:04에서 끝난 것을 감안하여 이어서 생성하거나 새 기준 시각 적용
    start_datetime = pd.to_datetime('2026-07-07 10:20:00')
    timestamps = [start_datetime + timedelta(seconds=5 * i) for i in range(n_records)]
    timestamps = pd.to_datetime(timestamps)
    
    # 2. 베이스라인 정상 데이터 생성 (sensor_data.csv 통계치 반영)
    # - 온도(Temperature_C): 평균 ~24.1°C, 표준편차 ~0.15°C
    # - 습도(Humidity_Percent): 평균 ~61.4%, 표준편차 ~0.3%
    # - 일산화탄소(CO_ppm): 평균 ~30.5 ppm, 표준편차 ~0.5 ppm
    temp_base = 24.1 + np.random.normal(0, 0.15, n_records)
    hum_base = 61.4 + np.random.normal(0, 0.3, n_records)
    co_base = 30.5 + np.random.normal(0, 0.5, n_records)
    
    temperature = temp_base.copy()
    humidity = hum_base.copy()
    co = co_base.copy()
    anomaly_label = np.zeros(n_records, dtype=int)
    
    # 3. 화재 징후(이상치) 시나리오 주입
    
    # [시나리오 1: 서서히 진행되는 과열/훈소 화재 징후 (Smoldering)]
    # - 구간: 2,000 ~ 2,500행 (약 41분)
    # - 특성: 온도가 서서히 45°C까지 상승, 습도는 35%로 하강, CO 농도는 120 ppm까지 서서히 상승
    s1_start, s1_end = 2000, 2500
    s1_len = s1_end - s1_start
    temperature[s1_start:s1_end] += np.linspace(0, 21, s1_len) + np.random.normal(0, 0.2, s1_len)
    humidity[s1_start:s1_end] -= np.linspace(0, 26, s1_len) + np.random.normal(0, 0.5, s1_len)
    co[s1_start:s1_end] += np.linspace(0, 90, s1_len) + np.random.normal(0, 2, s1_len)
    anomaly_label[s1_start:s1_end] = 1
    
    # [시나리오 2: 급격한 화염 확산 및 연소 (Flashover/Flame)]
    # - 구간: 6,000 ~ 6,300행 (약 25분)
    # - 특성: 온도가 85°C까지 급격히 상승, 습도는 15%로 급락, CO 농도는 450 ppm까지 폭발적으로 증가
    s2_start, s2_end = 6000, 6300
    s2_len = s2_end - s2_start
    temperature[s2_start:s2_end] += np.linspace(0, 61, s2_len) + np.random.normal(0, 0.5, s2_len)
    humidity[s2_start:s2_end] -= np.linspace(0, 46, s2_len) + np.random.normal(0, 0.8, s2_len)
    co[s2_start:s2_end] += np.linspace(0, 420, s2_len) + np.random.normal(0, 5, s2_len)
    anomaly_label[s2_start:s2_end] = 1
    
    # [시나리오 3: 돌발성 센서 노이즈/이상 스파이크 주입 (실무/전처리 실습용)]
    # - 단발성으로 비정상적인 극단값을 가지는 이상 데이터 주입 (라벨 1 부여)
    spike_indices = [1000, 3500, 4800, 7500, 9000]
    for idx in spike_indices:
        temperature[idx] = 110.0
        humidity[idx] = 8.0
        co[idx] = 600.0
        anomaly_label[idx] = 1
        
    # 물리적 최솟값 보정 (습도는 0% 이하로 갈 수 없으므로 제한)
    humidity = np.clip(humidity, 5.0, 100.0)
    co = np.clip(co, 0.0, None)
    
    # 4. 데이터프레임 빌드 및 저장
    df = pd.DataFrame({
        'Date': timestamps.strftime('%Y-%m-%d'),
        'Time': timestamps.strftime('%H:%M:%S'),
        'Temperature_C': np.round(temperature, 2),
        'Humidity_Percent': np.round(humidity, 1),
        'CO_ppm': np.round(co, 2),
        'Anomaly_Label': anomaly_label
    })
    
    df.to_csv(output_path, index=False)
    print(f"가상 데이터 생성 완료! 파일 저장 경로: {output_path}")
    print(f"전체 레코드 수: {len(df)}건")
    print(f"정상 데이터: {len(df[df['Anomaly_Label'] == 0])}건")
    print(f"화재/이상 데이터: {len(df[df['Anomaly_Label'] == 1])}건")

if __name__ == '__main__':
    generate_sensor_data()
