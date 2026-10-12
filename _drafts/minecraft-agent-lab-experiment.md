---
layout: single
post_type: experiment
title: "Minecraft Agent Lab 실험 기록: Qwen·Kev의 작업 중단과 재개"
date: 2026-10-12 00:00:00 +0900
last_modified_at: 2026-10-12
categories: [AI]
tags: [기술정리, 실험기록, Minecraft, Mineflayer, Qwen, Kev, DecisionModel, AIAgent]
permalink: /ai/minecraft-agent-lab-experiment/
excerpt: "Qwen·Kev로 채집·제작·귀가를 연결한 실행 조건과 결과, 전투 중단·재개 구조, 크리퍼 피해와 남은 과제를 기록한다."
toc: true
toc_label: "이 글의 내용"
draft: true
---

Minecraft Agent Lab은 Qwen3.8-27B의 계획, Kev-4B의 행동 선택, Mineflayer의 실행, 로컬 피해 반응 코드로 구성했다. 개발 실행에서 야간 채집·제작·귀가와 전투 후 작업 재개를 관찰했다. 실행 환경, 모델·코드 버전, 처리 순서, 관찰 결과와 제약조건을 기록한다.

기준 코드는 [7be99fa 버전](https://github.com/LeeJeaHyuk/minecraft-agent-lab/tree/7be99faac450f3031798a51185bd7d36ff79c958)이다. 도입 배경은 [게임 생각 글](/ai/minecraft-agent-lab-qwen-kev-gameplay/), 질문 유형과 확률 정의는 [Jev·Kev 개념 정리](/ai/jev-system-one-decision-model/)에서 다룬다.

## 사용한 모델과 역할

행동 선택에는 Jared Palmer의 공개 구현인 Kev를 사용했다. Kev와 TypeSafe의 Jev는 별도 모델이다. [Kev 프로젝트](https://github.com/jaredpalmer/kev)

Minecraft Agent Lab에서는 다음 모델을 사용했다.

| 모델 | 역할 | 기록된 실행 구성 |
| --- | --- | --- |
| Qwen3.8-27B | 목표·작업 순서 계획 | GGUF, 요청 수준 thinking 비활성화 |
| jaredpalmer/kev-4b | 행동 선택 | 공식 adapter/head, BF16, 추가 양자화 없음 |
| Qwen/Qwen3.5-4B-Base | Kev의 기반 모델 | Kev 구성에 포함 |

기록에 사용한 모델 파일과 revision은 다음과 같다. [Lab의 모델 기록](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#models)

| 구성 | 모델 파일·저장소 | revision |
| --- | --- | --- |
| 계획 모델 | `hugobugo34/Qwen3.8-27B-ROCmFPX-GGUF`의 `Qwen3.8-27B-ROCMFP4-COHERENT.gguf` | `27b12de0c9cb053f448fdea885713d8802ccbc3b` |
| 행동 선택 모델 | `jaredpalmer/kev-4b` | `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` |
| Kev 기반 모델 | `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |

## 시스템 구성

| 기능 | 출력·처리 내용 | 구현 |
| --- | --- | --- |
| 목표를 순서 있는 하위 작업으로 만들기 | 목표, 재료, 개수, 작업 순서 | QwenPlanner |
| 현재 하위 작업에서 다음 행동 고르기 | 허용된 행동 ID와 확률 분포 | KevPolicy |
| 이동·채집·제작·전투를 실행하기 | 게임 상태의 변화와 실행 결과 | Mineflayer 기반 환경 코드 |
| 피해를 받았을 때 진행 중인 동작 중단하기 | 즉시 중단과 방어 동작 | 로컬 damage reflex |

계획과 행동 선택은 각각 QwenPlanner와 KevPolicy가 처리한다. 피해 발생 시 최초 중단은 로컬 코드가 처리한다. [계획 실행 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/scripts/planned-agent.mjs), [피해 반응 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/damage-reflex.mjs)

`KevPolicy`는 `choice`와 `probabilities`를 사용하고, 선택된 행동 ID·확률 범위·확률 합계를 검사한다. `confidence`에 따른 자동 승격·보류 정책은 구현돼 있지 않다. [정책 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/policies/index.mjs)

## 실행 환경

| 구성 | 실행 조건 |
| --- | --- |
| 게임 | Minecraft Java 1.21.1, Survival, Normal 난이도 |
| 런타임 요구사항 | Node.js 22+, Java 21 |
| 게임 연동 라이브러리 | Mineflayer 4.37.1 |
| 계획 모델 | Qwen3.8-27B, `qwen3.8-27b`로 제공되는 OpenAI-compatible endpoint |
| 행동 선택 모델 | jaredpalmer/kev-4b, 공식 adapter/head, BF16 |
| 모델이 받는 정보 | 체력·허기·재고·장비·주변 엔티티·블록·집 상태와 행동 후보 |
| 관찰 방식 | Mineflayer가 읽은 구조화된 상태, 모델 입력으로 스크린샷을 사용하지 않음 |

Mineflayer가 게임 상태를 읽어 구조화된 관찰값을 구성한다. 모델의 입력은 상태 필드와 행동 후보이며, 스크린샷 입력은 사용하지 않았다. [README의 Setup·Models](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md), [환경 어댑터](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/adapter.mjs)

Mineflayer 기반 환경 코드는 허용된 행동 ID에 대응하는 이동·채집·제작·블록 설치 동작을 실행한다. 실행 후 재고와 장착 상태를 읽어 결과를 확인한다.

## 관찰 결과: 야간 채집·전투·귀가

야간 채집과 자연 발생 전투·작업 재개는 서로 다른 개발 실행이다. 같은 조건에서 반복한 시험으로 묶지 않고 각 실행에서 관찰한 결과를 나눠 기록한다.

| 개발 실행 사례 | 관찰된 결과 | 예외·제약조건 |
| --- | --- | --- |
| Normal 난이도 야간 채집 | 참나무 원목 2개를 수집하고 판자가 2개에서 10개로 증가했다. 문이 닫힌 집으로 돌아왔고, 이 실행에서는 추가 체력 손실이 보고되지 않았다. | 채집 행동 하나는 아이템 획득 뒤 타임아웃됐다. 실행기는 실제 재고를 보고 계속 진행했다. |
| 자연 발생 야간 전투 | 거미와 좀비 두 마리에 대응했다. 좀비 대응 이후에는 중단됐던 귀가를 재개하고, 문을 닫고, 수집한 음식을 먹었다. | 중단된 사냥의 완료 판정 오류를 발견한 뒤 수정했다. |
| 이후 크리퍼 피해 | 체력이 6.63/20까지 감소했다. | 크리퍼 회피는 미해결 과제로 기록돼 있다. |

이 결과와 제한 사항은 [공개된 Observed results](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#observed-results-with-qwen38--kev-4b)에 근거한다. 반복 횟수와 단계별 시간·전체 성공률은 이 결과표에 포함돼 있지 않다.

## 채집·제작 실행 순서

채집·제작용 실행기인 `scripts/planned-agent.mjs`는 다음 순서로 동작한다.

1. 현재 상태를 읽고 Qwen에게 목표의 작업 순서를 요청한다.
2. 계획에 나온 아이템 이름이 게임의 실제 capability 목록, 즉 환경이 제공하는 아이템 이름 목록에 있는지 확인한다.
3. 현재 단계가 이미 완료됐는지 실제 재고로 검사한다.
4. 환경에서 현재 행동 후보를 받고, 현재 단계에 맞는 후보로 좁힌다.
5. Kev에게 상태와 후보를 보내 다음 행동을 선택하게 한다.
6. 환경 API로 행동을 실행한 뒤 다시 관찰한다.

위 순서는 [채집·제작 실행기](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/scripts/planned-agent.mjs)에 해당한다. 집 짓기는 `scripts/build-agent.mjs`, 보관은 `scripts/storage-agent.mjs`, 장비 준비는 `scripts/equipment-agent.mjs`가 처리한다.

계획의 `count`는 단계 완료 시 필요한 최소 재고이다. 판자 8개가 목표라면 실제 `oak_planks` 재고가 8개 이상인지 검사한다. 재료가 이미 있으면 수집 단계를 건너뛸 수 있다. 제작으로 재료가 소비되므로 완료 여부는 각 단계의 현재 재고를 기준으로 검사한다. [계획 계약](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/planners/qwen.mjs), [완료 판정](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/planners/task-runner.mjs)

Qwen의 계획 갱신을 기다리는 동안에도 현재 수락한 계획으로 Kev의 행동 선택을 진행할 수 있다. 갱신된 답이 늦게 도착했는데 이미 다음 단계로 넘어갔다면, 실행기는 오래된 답을 버린다. 계획의 표현을 갱신하더라도 수락한 아이템·개수·순서를 바꾸지 못하도록 검사한다. 이 부분도 `TaskPlanner`에서 확인할 수 있다.

## 피해 대응과 작업 재개

Minecraft의 체력이 줄면 `installDamageReflex`는 이동 목표와 채굴을 중단하고 조작 상태를 해제한다. 방패가 실제 보조 손 슬롯에 있다면 방어를 활성화한다. **이 최초 중단은 Qwen이나 Kev의 응답을 기다리지 않는 로컬 코드이다.** [damage-reflex.mjs](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/damage-reflex.mjs)

즉시 중단한 뒤에 Kev의 판단이 이어진다. 위협 대응 루프에서는 현재 체력, 장비, 주변 위협과 허용된 대응 행동을 Kev에게 보낸다. 선택된 공격·후퇴·귀가 동작을 실행하고, 위협이 해소됐는지 다시 관찰한다. 안전 대응이 해결되지 않거나 낮은 체력으로 대피한 상태라면 작업을 무조건 재개하지 않는다. [task-safety.mjs](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/task-safety.mjs)

위협이 해소된 뒤에도 이전 행동을 모두 재생하지는 않는다. 이동·귀가 등 재개 가능한 행동은 현재 후보를 다시 확인한다. 제작이나 아이템 전송처럼 중복 실행의 영향이 있는 행동은 `action_needs_replan`으로 반환할 수 있다. 모든 실행기가 이 경우를 자동으로 다시 계획하는 것은 아니다. [행동 중단과 재개 구현](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/adapter.mjs)

## 다른 과제에서 확인한 결과

야간 플레이 외에도 작은 집과 출입문·조명, 상자 보관, 철 장비 준비가 개발 실행에서 관찰됐다. 집에는 문과 횃불 네 개를 설치하고 내부 귀환과 출입구 폐쇄를 확인했다. 상자는 내용물을 다시 읽었고, 철 장비는 곡괭이·검·방패·갑옷 네 부위를 준비한 뒤 실제 장착 슬롯을 확인했다. [공개 결과표](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#observed-results-with-qwen38--kev-4b)

준비 과정에는 사망 후 장비 회수와 운영자의 낮 시간 조정·주변 적 제거가 포함됐다. 일부 과제는 기존 집·상자와 저장된 세계 상태에 의존한다. 새 세계에서 시작한 무개입 생존 결과와 Ender Dragon 처치 결과는 보고되지 않았다.

## 로그와 검증 범위

환경 API는 상태·행동 후보·선택된 행동·확률·실행 결과·다음 상태를 JSONL로 기록한다. 실행·에피소드 ID, 세계 seed, 목표·하위 목표와 모델 정보도 함께 저장한다. [기록 생성 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/api.mjs), [JSONL 기록 형식](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/logging/trajectory.mjs)

| 시간 항목 | 로그 필드 | 기록값 |
| --- | --- | --- |
| 계획 | `planner_latency_ms` | 계획기가 전달한 지연 시간 |
| 행동 선택 | `decision_latency_ms` | 정책이 전달한 판단 지연 시간 |
| 행동 실행 | `action_duration_ms` | 환경의 행동 실행 결과에 포함된 소요 시간 |

계획·판단 지연 값이 전달되지 않은 경우 해당 필드는 `null`이다. 공개 결과 요약에는 기존 방식과 비교한 판단 시간이나 작업 전체 시간의 개선량이 제시돼 있지 않다.

게임 채팅은 Qwen의 공개 진행 발언, Kev 선택에 대응한 메시지, 로컬 시스템 중단 알림으로 출처를 표시한다. OBS 녹화는 선택 기능이다. 실행기는 직접 시작한 녹화만 종료하고, 다른 프로그램이 시작한 녹화는 유지한다. [채팅 출처 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/planners/chat-source.mjs), [녹화 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/recording/obs.mjs)

공개 README는 해당 코드 버전에서 자동 테스트 62개가 통과했다고 기록한다. 검증 대상은 계약과 회귀이다. 반복 플레이의 성공률은 제시돼 있지 않다.

공개 저장소에는 코드와 결과 요약이 있다. 원본 플레이 로그·녹화·세계 데이터는 공개돼 있지 않아 기록된 개별 실행의 초기 상태와 운영자 개입을 독립적으로 대조할 자료는 제공되지 않는다. [프로젝트 공개 범위](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/PRIVACY.md)

## 미해결 과제

기준 코드 버전의 미해결 과제는 크리퍼 회피, 식량 비축, 길찾기, 채집 후 아이템 회수 확인이다. 야간 생존 중 크리퍼 피해가 발생했고, 일부 채집은 아이템 획득 뒤 타임아웃됐다. [Known limitations](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#known-limitations)

## 실행 환경 준비와 실행 방법

기록에 사용한 코드 버전으로 실행 환경을 준비한다. 모델 가중치와 추론 서버는 별도 준비가 필요하며, Java 서버를 처음 만들 때는 Minecraft EULA를 직접 확인하고 동의 여부를 결정해야 한다. [프로젝트 설치 안내](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#setup)

```bash
git clone https://github.com/LeeJeaHyuk/minecraft-agent-lab.git
cd minecraft-agent-lab
git checkout 7be99faac450f3031798a51185bd7d36ff79c958
npm ci
cp .env.example .env
node --env-file=.env scripts/prepare-server.mjs
```

EULA 확인 후 서버 준비를 마쳤다면, 별도 터미널에서 게임 서버와 환경 API를 각각 실행한다.

```bash
# 게임 서버
npm run server

# 별도 터미널: Mineflayer 환경 API
npm start
```

`.env`의 `QWEN_BASE_URL`, `QWEN_MODEL`, `KEV_BASE_URL`, `KEV_REVISION`을 실제 추론 구성에 맞게 설정한다. Kev의 다운로드·시작 스크립트와 고정된 vendor 코드 버전은 [README의 Models 절](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#models)에 있다. `.env.example`은 설정 예제이며, 모델 가중치와 추론 서버는 별도로 준비한다.

게임과 두 모델이 준비된 뒤 아래 목표부터 확인할 수 있다.

```bash
npm run qwen:smoke
npm run agent -- kev 5
npm run planned -- "Collect two oak logs, then craft eight oak planks" 20
```

Minecraft Java 1.21.1 클라이언트로 `localhost:25565`에 접속하면 `LabBot`을 관찰할 수 있다. 공개 기본 설정은 loopback과 offline 인증을 사용하는 로컬 실험용이다. 해당 설정을 인터넷에 그대로 노출해서는 안 된다. [실행 설정](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/.env.example), [서버 사용 범위](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#setup)

위 명령의 목표는 원목 채집과 판자 제작이다. 집·상자·장비·야간 대응은 별도 실행기로 처리하며, 일부 과제는 앞선 준비 상태에 의존한다.

관련 생각 글: [Decision Model (2): 게임을 더 살아 있게 만들 수 있을까](/ai/minecraft-agent-lab-qwen-kev-gameplay/).
