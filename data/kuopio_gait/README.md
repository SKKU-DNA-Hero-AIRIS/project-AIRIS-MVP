# Kuopio Gait Dataset

## 연구 내 역할

영상 기반 OpenPose 관절 좌표로 계산한 보행지표를 3D 모션캡처, IMU, Force plate 데이터와 비교하여 정확도를 검증하는 데 사용합니다.

## 공식 자료

- 데이터 다운로드: https://zenodo.org/records/10559504
- 설명 논문 DOI: https://doi.org/10.1016/j.dib.2024.110841

## 주요 구성

- OpenPose BODY_25 관절 좌표
- 3D 모션캡처 마커
- Force plate
- IMU
- 참가자 메타데이터
- 보행속도별 반복 측정

## 초기 다운로드 권장 파일

- `readme.txt`
- `info_participants.xlsx`
- `measurement_data_18_to_34.zip`

18번부터 34번 참가자 묶음은 주요 누락 참가자가 명시되지 않아 초기 검증용으로 우선 활용합니다.

## AIRIS에서 사용할 항목

- OpenPose 관절 좌표
- MoCap 기준 좌표
- 발 접촉 시점
- 보행주기
- 보폭
- 보행속도
- 골반 및 하지 움직임
- IMU 비교 결과

## 주의사항

- `.mtb` 형식의 원시 IMU 파일보다 변환된 `imu_extracted`를 우선 활용
- 원본 ZIP과 `.mtb` 파일은 GitHub에 업로드하지 않음
- 원본 데이터는 `data/raw/kuopio_gait/`에 로컬로 저장
