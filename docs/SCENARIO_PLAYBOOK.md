# Scenario Playbook — 10개 맥락 시나리오 실행 결과

> 판단 계층의 설계는 [CONTROL_LOGIC.md](CONTROL_LOGIC.md)에 있다. 이 문서는 **실제 실행 결과**만 다룬다.
>
> 아래 모든 수치는 `python tools/run_scenarios.py`의 출력이며 [`assets/reports/scenario_decisions.json`](../assets/reports/scenario_decisions.json)에 원본이 저장되어 있다. 손으로 채운 값은 없다.

---

## 1. 요약

| ID | 시나리오 | 시설 | 고정 규칙 | ML (신뢰도) | 하한/상한 | **최종** | 결정 주체 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 사무실 정상 출입 | office | LOW | LOW (98%) | LOW / LOW | **LOW** | ML |
| S2 | 헬스장 운동 직후 | gym | HIGH | MEDIUM (78%) | LOW / MEDIUM | **MEDIUM** | ML |
| S3 | 병원 로비 *(S2와 생체 동일)* | hospital | HIGH | HIGH (100%) | LOW / HIGH | **HIGH** | ML |
| S4 | 어린이집 발열 아동 | daycare | HIGH | HIGH (100%) | HIGH / HIGH | **HIGH** | ML |
| S5 | 학교 하교 시간 고농도 미세먼지 | school | MEDIUM | HIGH (100%) | HIGH / HIGH | **HIGH** | ML |
| S6 | 공장 고온다습 작업 후 | factory | HIGH | HIGH (100%) | LOW / HIGH | **HIGH** | ML |
| S7 | 사무실 무증상 저산소 | office | HIGH | LOW (56%) | HIGH / HIGH | **HIGH** | `RULE_FLOOR` |
| S8 | 사무실 점심시간 미착용 밀집 | office | MEDIUM | MEDIUM (57%) | MEDIUM / HIGH | **MEDIUM** | ML |
| S9 | 어린이집 한가한 아침 | daycare | LOW | MEDIUM (53%) | LOW / LOW | **LOW** | `RULE_CEILING` |
| S10 | 헬스장 고강도 운동 직후 | gym | HIGH | HIGH (99%) | LOW / MEDIUM | **MEDIUM** | `RULE_CEILING` |

10건 중 7건은 ML 판단이 그대로 채택되었고, 3건에서 규칙이 개입했다(S7 상향, S9·S10 완화). 고정 규칙 기준선은 10건 중 **3건**에서 하이브리드와 다른 결론을 냈다 — S2·S10은 과잉 제어, S5는 과소 제어다.

---

## 2. 핵심 대조 — S2 vs S3

연구 질문의 주 가설을 직접 검증하는 쌍이다. 두 Context JSON의 `health_context`는 **완전히 동일**하다.

```json
{ "heart_rate": 138, "spo2": 98, "body_temperature": 37.1, "stress_index": 74 }
```

`vision_context`와 `environment_context`도 동일하다. **`facility_context`만 다르다.**

| | S2 (헬스장) | S3 (병원) |
| --- | --- | --- |
| `facility_type` | gym | hospital |
| `sensitivity` | LOW | HIGH |
| `vulnerable_population` | false | true |
| → `hr_x_sensitivity` | **0.00** | **0.80** |
| → `activity_plausible` | **0.69** | **0.00** |
| ML 확률 분포 | LOW 12.8% / **MEDIUM 77.5%** / HIGH 9.7% | LOW 0.0% / MEDIUM 0.1% / **HIGH 99.9%** |
| 최종 제어 | **MEDIUM** | **HIGH** |
| 액추에이션 | fan 65%, hold 5s, alert DASHBOARD | fan 100%, recirc CLOSED, hold 11s, alert MANAGER |
| 고정 규칙 기준선 | HIGH ❌ | HIGH ✅ |

S3의 최상위 ML 기여도는 `hr_x_sensitivity`가 **+4.16**이다. 즉 모델이 "심박수 138"을 보고 HIGH를 판단한 것이 아니라, **"고감도 시설에서의 심박수 138"** 이라는 교차 특징을 보고 판단했다. S2에서는 같은 특징이 0.00이 되고, 대신 `activity_plausible = 0.69`가 활성화된다.

고정 규칙 기준선(`heart_rate > 120 → HIGH`)은 두 상황을 **구분하지 못한다.** 원시 센서값만 보기 때문이다. S2에서 불필요하게 게이트를 최대로 가동하며, 이것이 README 1절이 지적한 문제의 실측 사례다.

---

## 3. 규칙이 ML을 뒤집은 사례

### S7 — 안전 하한이 ML의 실패를 잡아낸다 (`RULE_FLOOR`)

```json
"health_context": { "heart_rate": 88, "spo2": 91, "body_temperature": 36.9, "stress_index": 30 },
"environment_context": { "temperature": 23.5, "humidity": 44, "pm25": 12, "air_quality": "GOOD" },
"vision_context": { "person_count": 3, "mask_rate": 0.67, "crowding_level": "LOW" }
```

산소포화도 91%를 제외한 **모든 지표가 정상**인 조용한 사무실이다.

| 항목 | 값 |
| --- | --- |
| ML 확률 분포 | **LOW 55.7%** / MEDIUM 44.0% / HIGH 0.3% |
| `spo2_deficit` 기여도 | 0.50 → **−2.84** (LOW를 반대) |
| 발동 규칙 | **F1** 저산소증 (`SpO₂ < 92`) → 하한 HIGH |
| 최종 | **HIGH** (`RULE_FLOOR`) |

이 시나리오가 하이브리드 구조의 존재 이유다. 모델은 저산소 신호를 놓치지 않았다 — `spo2_deficit`의 기여도 −2.84는 LOW에 명확히 반대하고 있다. 그런데도 정상 범위인 나머지 15개 특징의 기여도 합이 이를 상쇄해 최종적으로 LOW가 55.7%로 승리했다. 확률 모델에서는 **단일 임계선 위반이 다수의 정상 신호에 희석될 수 있다.**

규칙 F1은 이 희석을 허용하지 않는다. 게이트는 HIGH로 동작하고, 판단 근거에는 `"산소포화도 91%로 저산소 임계선(92%) 미만"`이 기록된다. 신뢰도 56%짜리 LOW 예측에 사람의 안전을 맡기지 않는다는 것이 설계 의도다.

### S10 — 완화 상한이 과잉 제어를 막는다 (`RULE_CEILING`)

```json
"health_context": { "heart_rate": 165, "spo2": 96, "body_temperature": 37.4, "stress_index": 92 },
"environment_context": { "temperature": 29.0, "humidity": 70, "pm25": 30, "air_quality": "MODERATE" },
"facility_context": { "facility_type": "gym", "sensitivity": "LOW", "vulnerable_population": false }
```

| 항목 | 값 |
| --- | --- |
| ML 확률 분포 | LOW 0.0% / MEDIUM 0.9% / **HIGH 99.1%** |
| 상위 기여도 | `hr_excess` 1.00 → **+2.50**, `stress` 0.92 → **+1.38** |
| 반대 기여도 | `activity_plausible` 0.67 → **−1.00** |
| 발동 규칙 | **C1** 신체 활동으로 설명 가능 → 상한 MEDIUM |
| 최종 | **MEDIUM** (`RULE_CEILING`) |

주목할 점은 ML이 `activity_plausible = 0.67`을 **인지했고** 그 기여도가 −1.00으로 HIGH에 반대했음에도, 심박(+2.50)과 스트레스(+1.38)의 합이 이를 압도해 99.1% 신뢰도로 HIGH를 예측했다는 것이다.

특징 공학만으로는 부족하다는 증거다. 여러 신호가 동시에 극단으로 갈 때 학습된 완화 경향은 압도된다. C1은 "산소·체온이 정상인 저민감도 시설의 심박 상승은 MEDIUM을 넘지 않는다"를 **보장**으로 만든다. 결과적으로 팬은 100%가 아니라 65%로, 경보는 MANAGER가 아니라 DASHBOARD로 내려간다.

### S9 — 시설 민감도만으로 등급을 올리지 않는다 (`RULE_CEILING`)

```json
"health_context": { "heart_rate": 92, "spo2": 98, "body_temperature": 36.6, "stress_index": 28 },
"facility_context": { "facility_type": "daycare", "sensitivity": "HIGH", "vulnerable_population": true }
```

| 항목 | 값 |
| --- | --- |
| ML 확률 분포 | LOW 47.0% / **MEDIUM 53.0%** / HIGH 0.0% |
| 발동 규칙 | **C2** 전 지표 정상 → 상한 LOW |
| 최종 | **LOW** (`RULE_CEILING`) |

모든 지표가 정상인데도 ML이 MEDIUM으로 기운 이유는 학습 루브릭이 고감도·취약계층 시설에 기본 가중치(`facility_sensitivity` 0.45 + `vulnerable_population` 0.15)를 부여하기 때문이다. 확률 격차는 53% 대 47%로 사실상 동점이다.

C2는 생체·환경·혼잡이 모두 정상 범위일 때 시설 속성만으로 등급을 올리는 것을 차단한다. 취약계층 시설이라고 상시 게이트를 가동하면 에너지 비용과 통행 지연이 발생하고, 무엇보다 경보가 상시화되면 운영자가 경보를 무시하기 시작한다.

---

## 4. 나머지 시나리오

### S1 — 사무실 정상 출입 (기준선)

ML LOW 98.5%, C2 발동, 최종 LOW. 세 방식(고정 규칙·ML·하이브리드)이 모두 일치한다. 정상 상황에서 하이브리드 구조가 불필요한 개입을 하지 않음을 확인하는 대조군이다. fan 30% / hold 2s / alert NONE.

### S4 — 어린이집 발열 아동

체온 38.2℃. ML HIGH 100%, 규칙 F2(고감도 시설 발열)도 하한 HIGH. **ML과 규칙이 독립적으로 같은 결론**에 도달한 사례다.

상위 기여도가 `fever` +3.70, `fever_x_vulnerable` +2.03, `hr_x_sensitivity` +1.42으로, 모델이 발열 그 자체와 "취약계층 시설에서의 발열"을 별개 근거로 사용했다. 고감도 시설이므로 액추에이션이 강화되어 hold 11s / 에어샤워 on.

동일한 38.2℃가 사무실이라면 F2(`sensitivity = HIGH` 요구)와 F3(38.5℃ 기준) 모두 발동하지 않아 하한은 LOW에 머문다. 규칙 계층 안에도 맥락이 들어 있다는 예다.

### S5 — 학교 하교 시간 고농도 미세먼지

생체 지표는 전부 정상(HR 88, SpO₂ 97, 36.9℃)이지만 PM2.5 88㎍/㎥ + 24명 밀집 + 마스크 착용률 29%다. ML HIGH 100%, 규칙 3개 동시 발동(F5 취약계층 대기질, F6 밀집 대기질, F7 미착용 밀집).

**고정 규칙 기준선은 MEDIUM을 반환한다.** 생체 임계선을 하나도 넘지 않았기 때문이다. 고정 규칙이 과잉 제어(S2)만 하는 것이 아니라 **과소 제어**도 한다는 반대 방향 증거다. 상위 기여도는 `crowding` +4.40, `pm25_x_crowding` +2.59, `pm25_load` +2.03으로 개인 생체가 아닌 집단·환경 맥락이 판단을 이끌었다.

### S6 — 공장 고온다습 작업 후

HR 121, 체온 37.6℃, 스트레스 84, 실내 34℃/82%. ML HIGH 99.95%, 규칙 미발동(하한 LOW / 상한 HIGH). **규칙이 침묵하고 ML만으로 결정된 유일한 HIGH 사례**다.

어떤 단일 임계선도 명확히 위반하지 않았다(SpO₂ 96, 체온 38℃ 미만, 밀집 MEDIUM). 여러 중간 강도 신호가 겹친 상황이며, 상위 기여도 `thermal_load` +1.79 / `fever` +1.22 / `stress` +1.10 / `humidity_load` +1.06이 고르게 분산되어 있다. 규칙 조합으로 표현하기 까다로운 이런 연속적 융합이 ML 계층이 담당하는 영역이다.

C1이 발동하지 않은 이유는 `sensitivity = MEDIUM`(공장)이기 때문이다. 헬스장이라면 MEDIUM으로 완화되었을 것이다.

### S8 — 사무실 점심시간 미착용 밀집

개인 생체는 정상이나 20명 밀집 + 마스크 착용률 25% + PM2.5 41㎍/㎥. ML MEDIUM 57.3%(HIGH 42.7%로 접전), 규칙 F6·F7이 하한 MEDIUM. ML과 하한이 일치해 최종 MEDIUM.

Vision 맥락 단독으로 제어 등급이 올라간 사례다. 상위 기여도 `unmasked_x_crowding` +0.35가 이를 반영한다. 개인 위험이 아니라 **집단 전파 위험**을 근거로 게이트가 반응했다.

---

## 5. 연구 질문 대응

| 질문 | 근거 시나리오 | 결과 |
| --- | --- | --- |
| **Q1** 서로 다른 데이터 소스를 하나의 표준 Context로 통합할 수 있는가 | 전체 10건 | Vision·Health·Environment·Facility 4개 소스가 단일 JSON으로 통합되고, 판단 계층은 그 JSON만 소비한다. S5·S8은 Vision·Environment 맥락만으로, S7은 Health 단독으로 제어가 결정되었다 |
| **Q2** 동일 센서값이 맥락에 따라 다른 제어를 낼 수 있는가 | **S2 / S3** | `health_context`가 동일한 두 상황이 MEDIUM과 HIGH로 갈렸다. 고정 규칙 기준선은 둘 다 HIGH로 처리했다. S4의 38.2℃도 시설에 따라 하한 규칙 발동 여부가 갈린다 |
| **Q3** 판단 로직을 크게 수정하지 않고 확장할 수 있는가 | S2 / S10 | `gym` 시설 유형 추가는 `FACILITIES` 딕셔너리에 4줄을 넣은 것이 전부다. 중재 로직·특징 정의·ML 코드는 변경되지 않았다. `from_dict`가 미지의 키를 무시하므로 신규 센서 필드를 JSON에 먼저 추가해도 파이프라인이 깨지지 않는다 |

---

## 6. 관측된 사실 정리

1. **고정 규칙은 양방향으로 틀린다.** S2·S10에서 과잉 제어(HIGH → 실제 MEDIUM), S5에서 과소 제어(MEDIUM → 실제 HIGH). 홀드아웃 2,000건에서 루브릭 재현율 70.1%.
2. **ML은 융합을 잘하지만 보장하지 않는다.** 루브릭 재현율 95.1%, macro F1 0.950, `LOW ↔ HIGH` 오분류 0건. 그러나 S7에서 저산소 사례를 신뢰도 56%로 LOW로 분류했다.
3. **규칙 개입은 예외적이다.** 홀드아웃 개입률 7.75%(하한 상향 3.50% / 상한 완화 4.25%). 규칙 계층은 상시 관문이 아니라 예외 처리기로 작동한다.
4. **개입 방향이 양쪽 모두 필요하다.** 상향(S7)과 완화(S9·S10)가 실제로 각각 발생했다. 안전 규칙만 두면 과잉 제어가 남고, 완화 규칙만 두면 안전 보장이 사라진다.
5. **특징 공학은 규칙을 대체하지 못한다.** `activity_plausible`이 있어도 S10에서 ML은 99% 신뢰도로 HIGH를 예측했다. 학습된 경향과 강제된 상한은 다른 종류의 보장이다.

---

## 7. 재현

```bash
python tools/train_control_model.py   # assets/policy/control_policy.json
python tools/run_scenarios.py         # assets/reports/scenario_decisions.json
python -m unittest discover -s tests  # 24 tests
```

`tests/test_control.py`가 이 문서의 핵심 주장을 회귀 테스트로 고정한다 — S2/S3의 등급 분기, S7의 하한 상향, S10의 상한 완화, 하한이 상한을 이기는 불변식.
