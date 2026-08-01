# Multi-sensor Human Gait Dataset

정식 명칭:

A multi-sensor and cross-session human gait dataset captured through an optical system and inertial measurement units

## 연구 내 역할

동일한 참가자를 서로 다른 날짜에 반복 측정한 데이터를 이용하여 보행지표의 측정 재현성과 개인 내 변화량을 검증하는 데 사용합니다.

## 공식 자료

- 데이터 다운로드: https://figshare.com/articles/dataset/14727231
- 설명 논문: https://www.nature.com/articles/s41597-022-01638-2
- 관련 코드: https://github.com/geisekss/motion_capture_analysis

## 주요 구성

- 건강한 성인 25명
- 서로 다른 두 날짜 측정
- 날짜별 반복 보행 trial
- 3D 모션캡처 마커
- IMU
- 스마트폰 센서

## AIRIS에서 사용할 항목

- 동일인 날짜 간 변화
- 보행속도
- 보폭
- 케이던스
- 좌우 균형
- 골반 및 몸통 움직임
- 측정 재현성
- 정상적인 일간 변동 범위

## 한계

- 산모 또는 산후 여성 데이터가 아님
- 장기간 일별·주별 회복 데이터가 아님
- RGB 게이트 영상 기반 모델 학습용으로는 제한적임

## 주의사항

원본 데이터는 `data/raw/multisensor_gait/`에 로컬로 저장하고 GitHub에는 업로드하지 않습니다.
