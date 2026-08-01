# Health&Gait

## 연구 내 역할

일반 영상에서 보행속도, 보폭, 케이던스, 지지시간 등의 보행지표를 추출하는 모델을 개발하는 데 사용합니다.

## 공식 자료

- 데이터 다운로드: https://zenodo.org/records/14039922
- 설명 논문: https://www.nature.com/articles/s41597-024-04327-4
- 관련 코드: https://github.com/AVAuco/healthgait

## 주요 구성

- 보행 영상
- AlphaPose 기반 2D 관절 좌표
- 실루엣
- DensePose
- Optical Flow
- OptoGait 및 MuscleLAB 측정 결과

## AIRIS에서 사용할 항목

- 보행속도
- 보폭 및 활보장
- 케이던스
- 한발 및 양발 지지시간
- 좌우 관절 움직임
- 어깨 및 골반 흔들림

## 다운로드 안내

전체 데이터는 약 26.8GB이며 여러 개의 분할 압축파일로 제공됩니다.

초기 구조 확인 단계에서는 `dataset_samples.zip`만 내려받아 확인합니다. 전체 데이터를 받을 경우 `Health_Gait.z01`부터 마지막 분할파일과 `Health_Gait.zip`을 모두 동일한 폴더에 저장해야 합니다.

## 주의사항

- 원본 데이터는 `data/raw/health_gait/`에 로컬로 저장
- 전체 ZIP 및 영상은 GitHub에 업로드하지 않음
- `sample/`에는 최소 크기의 구조 예시만 저장
