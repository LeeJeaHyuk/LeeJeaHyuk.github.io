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

Minecraft Agent Lab에서는 Qwen3.8-27B가 계획을 세우고 Kev-4B가 현재 행동을 선택하도록 구성했다. 밤에 원목을 모아 판자를 만들고 집으로 돌아오는 실행, 전투 뒤 중단된 귀가를 이어가는 실행을 확인했다. 이 글은 그 환경과 관찰 결과를 정리하고, 결과에 영향을 준 실행 코드와 실패 조건을 함께 설명한다.

코드는 [7be99fa 버전](https://github.com/LeeJeaHyuk/minecraft-agent-lab/tree/7be99faac450f3031798a51185bd7d36ff79c958)을 기준으로 읽는다. 선택의 배경은 [게임 생각 글](/ai/minecraft-agent-lab-qwen-kev-gameplay/), 질문과 확률의 개념은 [Jev·Kev 개념 정리](/ai/jev-system-one-decision-model/)에 연결했다.

## 사용한 모델과 역할

Jev는 TypeSafe의 모델이다. Kev는 Jared Palmer가 공개한 별도의 Jev 유사 결정 모델 계열이며, 직접 실행하거나 학습할 수 있도록 공개한 구현이다. 두 이름이 비슷하고 API 형태가 닮았더라도 같은 가중치나 같은 성능을 가진 모델이라는 뜻은 아니다. [Kev 프로젝트](https://github.com/jaredpalmer/kev)

Minecraft Agent Lab에서는 다음 모델을 사용했다.

| 모델 | 역할 | 기록된 실행 구성 |
| --- | --- | --- |
| Qwen3.8-27B | 장기 목표와 하위 작업 계획 | OpenAI-compatible endpoint, GGUF 변형, 요청 수준 thinking 비활성화 |
| jaredpalmer/kev-4b | 제한된 행동 후보 중 선택 | 공식 adapter/head, BF16, 추가 양자화 없음 |
| Qwen/Qwen3.5-4B-Base | Kev의 기반 모델 | Kev 구성에 포함되며 별도 계획 모델은 아님 |

이 구성과 고정된 모델 revision은 [Lab의 모델 기록](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#models)에 명시돼 있다. Jev의 공개 자료를 읽으며 관심을 가진 ‘상황을 보고 제한된 결정을 반환한다’는 방식을, Lab에서는 Kev와 Qwen의 역할 분리로 살펴볼 수 있다.

모델과 실행 코드를 함께 보면 역할 분리가 더 분명해진다.

| 맡길 일 | 필요한 출력 | 이 프로젝트의 담당 |
| --- | --- | --- |
| 목표를 순서 있는 하위 작업으로 만들기 | 목표, 재료, 개수, 작업 순서 | QwenPlanner |
| 현재 하위 작업에서 다음 행동 고르기 | 허용된 행동 ID와 확률 분포 | KevPolicy |
| 이동·채집·제작·전투를 실행하기 | 게임 상태의 변화와 실행 결과 | Mineflayer 기반 환경 코드 |
| 피해를 받았을 때 진행 중인 동작 중단하기 | 즉시 중단과 방어 동작 | 로컬 damage reflex |

이 역할 구분은 [계획 실행 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/scripts/planned-agent.mjs)와 [피해 반응 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/damage-reflex.mjs)에서 확인할 수 있다. 피해 반응처럼 즉시 처리할 일은 로컬 코드가 맡고, 계획과 행동 선택은 별도 모델이 맡는다.

현재 Lab의 `KevPolicy`는 선택된 행동 ID, 확률의 범위, 확률 합계를 검사한다. Jev 문서의 `confidence`에 따른 자동 승격·보류 정책을 구현한 것으로 읽으면 안 된다. 해당 코드가 주로 사용하는 것은 `choice`와 `probabilities`이다. [정책 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/policies/index.mjs)

## 어떤 환경에서 플레이했나?

| 구성 | 실행 조건 |
| --- | --- |
| 게임 | Minecraft Java 1.21.1, Survival, Normal 난이도 |
| 실행 코드 | Node.js 22+, Java 21, Mineflayer 4.37.1 |
| 계획 모델 | Qwen3.8-27B, `qwen3.8-27b`로 제공되는 OpenAI-compatible endpoint |
| 행동 선택 모델 | jaredpalmer/kev-4b, 공식 adapter/head, BF16 |
| 모델이 받는 정보 | 체력·허기·재고·장비·주변 엔티티·블록·집 상태와 행동 후보 |
| 관찰 방식 | Mineflayer가 읽은 구조화된 상태, 모델 입력으로 스크린샷을 사용하지 않음 |

환경과 모델 구성은 [README의 Setup·Models](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md), 상태 필드는 [환경 어댑터](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/adapter.mjs)에서 확인했다. 이 관찰 방식은 사람이 보는 화면을 그대로 읽는 방식과 관측 조건이 다르므로, 화면 기반 에이전트와 비교할 때도 그 차이를 밝혀야 한다.

위 구성에서 Mineflayer 기반 환경 코드는 이동·채집·제작·블록 설치 같은 동작을 실행하고, 재고와 장착 상태를 읽어 결과를 확인한다. 모델이 임의의 JavaScript나 게임 명령을 생성해 실행하는 구조가 아니다.

## 플레이 기록: 밤에 채집하고 집으로 돌아오기

야간 채집과 자연 발생 전투·작업 재개는 서로 다른 개발 실행이다. 같은 조건에서 반복한 시험으로 묶지 않고 각 실행에서 관찰한 결과를 나눠 기록한다.

| 개발 실행 사례 | 관찰된 결과 | 함께 기록해야 할 조건 |
| --- | --- | --- |
| Normal 난이도 야간 채집 | 참나무 원목 2개를 수집하고 판자가 2개에서 10개로 증가했다. 문이 닫힌 집으로 돌아왔고, 이 실행에서는 추가 체력 손실이 보고되지 않았다. | 채집 행동 하나는 아이템 획득 뒤 타임아웃됐다. 실행기는 실제 재고를 보고 계속 진행했다. |
| 자연 발생 야간 전투 | 거미와 좀비 두 마리에 대응했다. 좀비 대응 이후에는 중단됐던 귀가를 재개하고, 문을 닫고, 수집한 음식을 먹었다. | 중단된 사냥의 완료 판정 문제를 발견한 뒤 수정했다. 모든 위협 대응이 안정적이었다는 뜻은 아니다. |
| 이후 크리퍼 피해 | 체력이 6.63/20까지 감소했다. | 일반적인 야간 생존의 안전성은 확보되지 않았다. |

이 결과와 제한 사항은 [공개된 Observed results](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#observed-results-with-qwen38--kev-4b)에 근거한다. 반복 횟수와 단계별 시간·전체 성공률은 이 결과표에 포함돼 있지 않다.

이 사례에서 실행 구조를 이해하는 데 필요한 부분은 위협 대응 뒤의 작업 재개이다. 채집 중 위험에 대응했다면 원래 목표와 실제로 얻은 재고를 다시 확인해야 한다. 공격이 끝났다는 이유로 이전 행동을 그대로 반복하면 제작이나 보관 같은 작업이 중복될 수 있다. 다음 절에서는 이 확인이 실행 순서에 어떻게 들어가는지 살펴보자.

## 목표에서 실제 행동까지 이어지는 과정

채집·제작용 실행기인 `scripts/planned-agent.mjs`는 다음 순서로 동작한다.

1. 현재 상태를 읽고 Qwen에게 목표의 작업 순서를 요청한다.
2. 계획에 나온 아이템 이름이 게임의 실제 capability 목록, 즉 환경이 제공하는 아이템 이름 목록에 있는지 확인한다.
3. 현재 단계가 이미 완료됐는지 실제 재고로 검사한다.
4. 환경에서 현재 행동 후보를 받고, 현재 단계에 맞는 후보로 좁힌다.
5. Kev에게 상태와 후보를 보내 다음 행동을 선택하게 한다.
6. 환경 API로 행동을 실행한 뒤 다시 관찰한다.

이 순서는 [실제 실행기](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/scripts/planned-agent.mjs)에서 확인할 수 있다. 집 짓기·보관·장비 준비에는 별도 실행기가 있으므로, 이 파일 하나가 모든 플레이 과제를 처리한다고 읽으면 안 된다.

계획의 `count`는 “이번에 추가로 얻을 개수”가 아니라 해당 단계가 끝났을 때 필요한 최소 재고이다. 예를 들어 판자 8개가 목표라면 실제 `oak_planks` 재고가 8개 이상인지 검사한다. 재료가 이미 있으면 불필요한 수집 단계를 건너뛸 수 있고, 제작으로 앞 단계의 재료가 소비되는 것도 계획에서 고려해야 한다. [계획 계약](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/planners/qwen.mjs), [완료 판정](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/planners/task-runner.mjs)

Qwen의 계획 갱신을 기다리는 동안에도 현재 수락한 계획으로 Kev의 행동 선택을 진행할 수 있다. 갱신된 답이 늦게 도착했는데 이미 다음 단계로 넘어갔다면, 실행기는 오래된 답을 버린다. 계획의 표현을 갱신하더라도 수락한 아이템·개수·순서를 바꾸지 못하도록 검사한다. 이 부분도 `TaskPlanner`에서 확인할 수 있다.

## 피해 대응은 모델 호출보다 먼저 시작한다

Minecraft의 체력이 줄면 `installDamageReflex`는 이동 목표와 채굴을 중단하고 조작 상태를 해제한다. 방패가 실제 보조 손 슬롯에 있다면 방어를 활성화한다. **이 최초 중단은 Qwen이나 Kev의 응답을 기다리지 않는 로컬 코드이다.** [damage-reflex.mjs](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/damage-reflex.mjs)

즉시 중단한 뒤에 Kev의 판단이 이어진다. 위협 대응 루프에서는 현재 체력, 장비, 주변 위협과 허용된 대응 행동을 Kev에게 보낸다. 선택된 공격·후퇴·귀가 동작을 실행하고, 위협이 해소됐는지 다시 관찰한다. 안전 대응이 해결되지 않거나 낮은 체력으로 대피한 상태라면 작업을 무조건 재개하지 않는다. [task-safety.mjs](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/task-safety.mjs)

위협이 해소된 뒤에도 이전 행동을 모두 재생하지는 않는다. 이동·귀가 등 재개 가능한 행동은 현재 후보를 다시 확인한다. 제작이나 아이템 전송처럼 중복 실행의 영향이 있는 행동은 `action_needs_replan`으로 반환할 수 있다. 모든 실행기가 이 경우를 자동으로 다시 계획하는 것은 아니다. [행동 중단과 재개 구현](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/adapter.mjs)

따라서 플레이 결과에는 모델의 판단뿐 아니라 게임 상태를 읽는 코드, 실행 스킬, 즉시 중단, 재개 조건도 함께 영향을 준다.

## 다른 과제에서 확인한 결과

야간 플레이 외에도 작은 집과 출입문·조명, 상자 보관, 철 장비 준비가 개발 실행에서 관찰됐다. 집에는 문과 횃불 네 개를 설치하고 내부 귀환과 출입구 폐쇄를 확인했다. 상자는 내용물을 다시 읽었고, 철 장비는 곡괭이·검·방패·갑옷 네 부위를 준비한 뒤 실제 장착 슬롯을 확인했다. [공개 결과표](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#observed-results-with-qwen38--kev-4b)

하지만 준비 과정에는 사망 후 장비 회수와 운영자의 낮 시간 조정·주변 적 제거가 포함됐다. 이것을 새 세계에서 시작한 무개입 생존이나 게임 클리어로 표현할 수는 없다. 이 프로젝트는 Ender Dragon 처치를 주장하지 않는다.

## 무엇을 기록해야 결과를 판단할 수 있을까?

환경 API는 모델에게 준 상태와 후보, 선택된 행동, 확률, 다음 상태를 기록한다. 계획 모델과 행동 선택 모델의 이름·revision·지연 시간도 구분한다. 모델이 행동을 결정하는 데 걸린 시간과 실제 이동·제작에 걸린 시간을 나눠 봐야 병목을 찾을 수 있다. 채집을 빠르게 선택해도 나무까지 이동하고 아이템을 줍는 데는 시간이 걸리기 때문이다. [기록 생성 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/api.mjs), [JSONL 기록 형식](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/logging/trajectory.mjs)

관찰용 게임 채팅에도 출처를 붙인다. Qwen이 만든 공개 진행 발언, Kev의 선택에 대응한 메시지, 로컬 시스템의 중단 알림을 구분한다. 이 발언을 모델의 비공개 추론 과정이라고 부르면 안 된다. OBS 녹화도 선택 기능이며, 다른 프로그램이 이미 시작한 녹화를 임의로 종료하지 않도록 소유권을 구분한다. [채팅 출처 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/planners/chat-source.mjs), [녹화 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/recording/obs.mjs)

공개 README는 이 스냅샷이 자동 테스트 62개를 통과했다고 기록한다. 테스트는 계약과 회귀를 확인하며, 플레이 성공률이나 일반적인 게임 능력의 증거와는 구분해야 한다.

공개 저장소에는 코드와 결과 요약이 있고 원본 플레이 로그·녹화·세계 데이터는 포함돼 있지 않다. 같은 조건을 대조하려면 이 자료와 운영자 개입 기록도 필요하다. [프로젝트 공개 범위](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/PRIVACY.md)

## 다음 실행에서 확인할 부분

현재 기록으로 확인할 수 있는 것은 특정 개발 실행에서 채집·제작·전투·귀가를 연결했다는 점이다. 크리퍼 회피, 식량 비축, 길찾기와 아이템 회수 확인은 여전히 개선이 필요한 부분으로 기록돼 있다.

반복 실험에서는 같은 목표와 고정된 모델 revision을 사용하고, 세계 조건·시작 재고·사망·운영자 개입을 함께 남겨야 한다. 특히 ‘전투 뒤 귀가를 재개한 비율’과 ‘채집 타임아웃 후 실제 획득을 확인한 경우’를 따로 보면, 모델의 선택 문제와 실행기의 판정 문제를 구분하는 데 도움이 될 것으로 생각한다. 다음 실행에서는 이 두 항목을 분리해 집계하려 한다.

## 직접 확인할 때의 실행 순서

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

`.env`의 `QWEN_BASE_URL`, `QWEN_MODEL`, `KEV_BASE_URL`, `KEV_REVISION`을 실제 추론 구성에 맞게 설정해야 한다. Kev의 다운로드·시작 스크립트는 공개 README의 Models 절에 설명돼 있다. 샘플 설정은 모델 서버가 준비돼 있다는 뜻이 아니다.

게임과 두 모델이 준비된 뒤 아래 목표부터 확인할 수 있다.

```bash
npm run qwen:smoke
npm run agent -- kev 5
npm run planned -- "Collect two oak logs, then craft eight oak planks" 20
```

Minecraft Java 1.21.1 클라이언트로 `localhost:25565`에 접속하면 `LabBot`을 관찰할 수 있다. 공개 기본 설정은 loopback과 offline 인증을 사용하는 로컬 실험용이다. 해당 설정을 인터넷에 그대로 노출해서는 안 된다. [실행 설정](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/.env.example), [서버 사용 범위](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#setup)

집·상자·장비·야간 대응을 보는 별도 실험은 앞선 준비 상태에 의존할 수 있다. 채집 예제를 실행했다고 집이나 장비 준비까지 자동으로 완료된다고 가정해서는 안 된다.

게임에 적용하며 기대와 생각이 바뀐 과정은 [Decision Model (2): 게임을 더 살아 있게 만들 수 있을까](/ai/minecraft-agent-lab-qwen-kev-gameplay/)에서 이어진다.
