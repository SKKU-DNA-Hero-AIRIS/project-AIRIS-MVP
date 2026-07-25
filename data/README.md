# Synthetic Biosignal & Gate Data

> **에어게이트 Context-Aware 파이프라인의 입력 데이터 생성 계층**

루트 `README.md`의 파이프라인에서 이 폴더는 **Raw Data → Context** 구간의 입력을
합성한다. 비전 파트(YOLO 인원·마스크 탐지)가 `vision_context`를 만든다면, 이
폴더는 그 짝인 **생체 신호**와 **게이트 통합 레코드**를 만든다.

실제 하드웨어·임상 데이터 없이 파이프라인과 방법론을 검증하기 위한 것이며,
"측정할 수 있다고 가정"하고 완벽한 값을 찍어내는 대신 **측정 채널의 물리적 한계를
명시적으로 모델링**하는 것을 설계 원칙으로 삼는다.

---

## 1. 3층 생성 구조

측정값을 진짜 생리값과 분리해서 만든다. `air_gate.py`가 세 단계를 거친다.

```text
1) 진짜 생리 상태        health_synth 의 잠재상태 모델
        ↓  환경이 생리에 실제로 영향
2) 환경 보정된 진짜 값    더위 → 심박↑,  PM2.5 → 호흡수↑
        ↓  게이트 측정 채널 열화
3) 게이트가 관측한 값     편향 + 잡음 + 실패(마스크·움직임·조명)
```

1~2단계 값은 레코드의 `_truth` 필드에 남겨 측정 오차의 영향을 직접 분석할 수 있게
한다. 배포 시에는 `_truth`를 제거한다.

---

## 2. 파일 구성

| 파일 | 역할 |
|------|------|
| [`health_synth.py`](health_synth.py) | 잠재상태 기반 생체징후 생성기 (Gaussian copula 상관 + 구조방정식, NEWS2 파생) |
| [`air_gate.py`](air_gate.py) | 시설별 게이트 멀티모달 레코드 생성기 (위 3층 구조) |
| [`ambient_correction.py`](ambient_correction.py) | 주변온도 보정 + 평시 대비 이상률 유의성 모니터링 |
| `*.jsonl`, `gate_persons_flat.csv` | 생성 산출물 — **커밋하지 않음**, 아래 명령으로 재생성 (`.gitignore` 처리) |

### 측정 채널의 한계 (반드시 인지)

- **체온** — 적외선은 **피부 표면온**을 잰다(심부온 아님). 주변온도 의존성이
  지배적 오차원이며, 32℃ 환경에서는 과대 측정한다.
- **심박** — rPPG로 RGB 영상에서 추정 가능하나, 마스크가 ROI를 가리고 보행·조명에
  취약하다.
- **SpO2** — ⚠️ 통과형 카메라/적외선으로는 사실상 측정 불가. 94% vs 97% 구분 용도엔
  부적합(오차 ±2~3%p).
- **stress_index** — HRV가 필요해 수 초 통과 시간으로는 신뢰 불가. 채널 제거 권장.

---

## 3. 실행 방법

```bash
pip install numpy pandas scipy
```

### 생체징후 표본 생성

```bash
# 단면 20,000행
python health_synth.py -n 20000 -o synthetic_vitals.csv

# 시계열 (300명 × 96스텝)
python health_synth.py --timeseries --subjects 300 --steps 96 -o vitals_ts.csv
```

### 게이트 레코드 생성

`--facility {hospital, school, office, cafe, transit}` 로 모집단이 바뀐다.

```bash
# 학교 · 평시
python air_gate.py -f school -o gate_school_spring.jsonl

# 병원 · 여름철 (주변온도 편향 +14)
python air_gate.py -f hospital --season-bias 14 -o gate_hospital_summer.jsonl

# 병원 · 감염 유행 (유병률 4배)
python air_gate.py -f hospital --outbreak 4.0 -o gate_hospital_outbreak.jsonl
```

---

## 4. 데이터 스키마

### 게이트 레코드 (`*.jsonl`, 프레임당 1줄)

```text
timestamp · gate_id · vision_context · health_context
environment_context · facility_context · persons[] · aggregate_signal
```

`persons[]`는 프레임 내 개인별 측정값과 `_truth`(진짜값)를 함께 담는다.

### 개인 평면화 CSV (`gate_persons_flat.csv`, 17열)

```text
timestamp, ambient_temp, pm25, crowding, masked, ir_bias, n_valid,
meas_temp, true_temp, meas_hr, true_hr, meas_spo2, true_spo2,
latent_state, is_abnormal, on_beta_blocker, is_elderly
```

`meas_*`(게이트 관측) vs `true_*`(진짜값) 쌍으로 측정 오차를 직접 분석한다.

---

## 5. 핵심 설계 결정

**라벨은 임계값이 아니라 잠재상태에서 나온다.** 임계값으로 라벨링하면 분류기가
임계값을 역산할 뿐이라 순환논리가 된다. severity가 편차 크기와 등급을 동시에
정해 등급 1은 정상과 겹치고 등급 3은 분리된다 — 겹침이 문제의 실제 난이도다.

**교란변수를 의도적으로 심는다.** 정상인데 비정상처럼(운동 직후·운동선수 서맥·
추운 환경), 비정상인데 정상처럼(베타차단제·고령자 발열 둔화).

**고정 체온 임계값은 작동하지 않는다.** 주변온도 일주기 드리프트가 유행 신호보다
크기 때문이다. `ambient_correction.py`의 실측 결과:

| 방식 | 평시 → 유행 | 배수 | 판정 |
|------|-------------|------|------|
| 고정 임계값 37.5℃ | 0.619 → 0.691 | 1.12 | 탐지 불가 |
| 주변온도 잔차화 | 0.050 → 0.130 | **2.60** | 전 시간대 유의 |

배포 시 평시 2~4주간 회귀계수를 학습하고 계절이 바뀌면 재학습한다.

---

## ⚠️ 의료 사용 금지

합성 데이터로 학습한 모델은 실제 건강 스크리닝에 사용할 수 없다. 특히 베타차단제·
고령자 위음성 분포는 합성으로 재현되지 않아, 실제 배포 시 정말 아픈 사람을 놓친다.
**방법론 연구·파이프라인 검증 전용**이다.
