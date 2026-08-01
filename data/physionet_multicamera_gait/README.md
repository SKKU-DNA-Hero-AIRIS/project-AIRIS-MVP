# PhysioNet Multi-camera Gait

정식 명칭:

A multi-camera and multimodal dataset for posture and gait analysis

## 연구 내 역할

카메라 위치, 촬영 장소, 보행속도 및 이동 경로가 달라져도 보행 분석 모델이 안정적으로 작동하는지 검증하는 데 사용합니다.

## 공식 자료

- 데이터 다운로드: https://physionet.org/content/multi-gait-posture/1.0.0/
- 설명 논문: https://www.nature.com/articles/s41597-022-01722-7
- 데이터 DOI: https://doi.org/10.13026/fyxw-n385

## 주요 구성

- 건강한 참가자 14명
- 세 가지 보행속도
- 세 가지 이동 경로
- 세 장소
- 두 대의 Depth 카메라
- 2D 및 3D 관절 좌표
- Xsens 기반 골격 기준값
- 카메라 보정정보

## 초기 다운로드 권장 항목

- `processed_data`
- `subject_metadata.csv`
- `processed_data_description.txt`
- `code`

## AIRIS에서 사용할 항목

- 카메라 위치 변화 대응
- 보행속도 변화 대응
- 이동 경로 변화 대응
- 2D 및 3D 관절 좌표 비교
- 실제 게이트 환경 적용 가능성 검증

## 한계

- 보행보조기 주변에서 수집된 데이터
- 참가자 수가 비교적 적음
- 임산부 또는 산후 여성 데이터가 아님

## 주의사항

원본 Depth 영상과 전체 데이터는 `data/raw/physionet_multicamera_gait/`에 로컬로 저장하고 GitHub에는 업로드하지 않습니다.
