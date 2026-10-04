---
layout: single
title: "Qwen과 Kev로 Minecraft를 플레이하기: 채집, 전투, 그리고 귀가"
date: 2026-10-04 00:00:00 +0900
categories: [AI]
tags: [Minecraft, Mineflayer, Qwen, Kev, Jev, AIAgent]
permalink: /ai/minecraft-agent-lab-qwen-kev-gameplay/
excerpt: "Minecraft Agent Lab의 공개 플레이 기록을 바탕으로 야간 채집, 위협 대응, 귀가 재개와 실제 상태 검증을 설명합니다."
toc: true
toc_label: "이 글의 내용"
draft: true
---

Minecraft Agent Lab에서는 Qwen이 작업 순서를 계획하고, Kev가 현재 가능한 행동을 고르며, Mineflayer가 게임 안에서 그 행동을 실행합니다. 플레이를 이어가는 데 필요한 것은 행동 선택만이 아니었습니다. 피해를 받았을 때 작업을 멈추고, 위협을 처리한 뒤, 바뀐 상태에서 작업을 이어갈 수 있어야 했습니다.

이 글은 [minecraft-agent-lab의 공개 스냅샷](https://github.com/LeeJeaHyuk/minecraft-agent-lab/tree/7be99faac450f3031798a51185bd7d36ff79c958)을 기준으로 작성했습니다. 앞 글에서 설명한 [Jev의 결정 모델 특성](/ai/jev-system-one-decision-model/)과 연결되는 사용 사례이지만, **이 실험이 직접 호출한 결정 모델은 Kev-4B이며 Jev 플레이 실험은 아닙니다.**

> 초안입니다. 플레이 결과는 프로젝트 README에 공개된 개발 실행의 관찰 기록을 바탕으로 정리했습니다. 코드 구조는 직접 확인했지만, 글 작성 과정에서 게임이나 모델을 다시 실행하지는 않았습니다. 원본 영상·세계 데이터·개발 로그는 공개 저장소에 포함돼 있지 않으므로, 아래 결과는 독립적으로 재현한 벤치마크가 아닙니다.

## 어떤 환경에서 플레이했나요?

| 구성 | 공개 기록에서 확인한 내용 |
| --- | --- |
| 게임 | Minecraft Java 1.21.1, Survival, Normal 난이도 |
| 실행 코드 | Node.js 22+, Java 21, Mineflayer 4.37.1 |
| 계획 모델 | Qwen3.8-27B, `qwen3.8-27b`로 제공되는 OpenAI-compatible endpoint |
| 행동 선택 모델 | jaredpalmer/kev-4b, 공식 adapter/head, BF16 |
| 모델이 받는 정보 | 체력·허기·재고·장비·주변 엔티티·블록·집 상태와 행동 후보 |
| 관찰 방식 | Mineflayer가 읽은 구조화된 상태, 모델 입력으로 스크린샷을 사용하지 않음 |

환경과 모델 구성은 [README의 Setup·Models](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md), 상태 필드는 [환경 어댑터](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/adapter.mjs)에서 확인했습니다. 이 관찰 방식은 사람이 보는 화면을 그대로 읽는 방식과 관측 조건이 다르므로, 화면 기반 에이전트와 비교할 때도 그 차이를 밝혀야 합니다.

핵심 구성은 세 부분입니다. Qwen은 목표를 하위 작업으로 나누고, Kev는 각 작업에서 허용되는 행동을 선택합니다. 이동 경로 계산과 제작·블록 설치 같은 구체적인 실행은 환경 코드가 맡습니다. 모델이 임의의 JavaScript나 게임 명령을 생성해 실행하는 구조가 아닙니다.

## 플레이 기록: 밤에 채집하고 집으로 돌아오기

공개된 결과 중에서는 야간 플레이가 이 구조의 필요성을 잘 보여줍니다. 다만 아래 장면들은 같은 조건의 반복 시험이나 하나의 연속 녹화에서 뽑은 타임라인이 아닙니다. **README에 기록된 야간 채집 실행과 자연 발생 전투·작업 재개 사례를 나눠 정리한 것입니다.**

| 개발 실행 사례 | 관찰된 결과 | 함께 기록해야 할 조건 |
| --- | --- | --- |
| Normal 난이도 야간 채집 | 참나무 원목 2개를 수집하고 판자가 2개에서 10개로 증가했습니다. 문이 닫힌 집으로 돌아왔고, 이 실행에서는 추가 체력 손실이 보고되지 않았습니다. | 채집 행동 하나는 아이템 획득 뒤 타임아웃됐습니다. 실행기는 실제 재고를 보고 계속 진행했습니다. |
| 자연 발생 야간 전투 | 거미와 좀비 두 마리에 대응했습니다. 좀비 대응 이후에는 중단됐던 귀가를 재개하고, 문을 닫고, 수집한 음식을 먹었습니다. | 중단된 사냥의 완료 판정 문제를 발견한 뒤 수정했습니다. 모든 위협 대응이 안정적이었다는 뜻은 아닙니다. |
| 이후 크리퍼 피해 | 체력이 6.63/20까지 감소했습니다. | 일반적인 야간 생존의 안전성은 확보되지 않았습니다. |

이 결과와 제한 사항은 [공개된 Observed results](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#observed-results-with-qwen38--kev-4b)에 근거합니다. 기록의 값은 그대로 사용했고, 단계별 시간이나 모델 지연·성공률은 추가로 만들어 넣지 않았습니다.

여기서 볼 부분은 “좀비를 이겼다”는 결과 하나보다 작업의 연결입니다. 채집 중 위험에 대응했다면, 그 뒤에는 원래 목표가 무엇이었는지와 실제로 무엇을 이미 얻었는지를 다시 확인해야 합니다. 공격이 끝났다고 이전 행동을 그대로 다시 실행하면 제작이나 보관 같은 작업이 중복될 수도 있습니다.

## 목표에서 실제 행동까지 이어지는 과정

채집·제작용 실행기인 `scripts/planned-agent.mjs`는 다음 순서로 동작합니다.

1. 현재 상태를 읽고 Qwen에게 목표의 작업 순서를 요청합니다.
2. 계획에 나온 아이템 이름이 게임의 실제 capability 목록에 있는지 확인합니다.
3. 현재 단계가 이미 완료됐는지 실제 재고로 검사합니다.
4. 환경에서 현재 행동 후보를 받고, 현재 단계에 맞는 후보로 좁힙니다.
5. Kev에게 상태와 후보를 보내 다음 행동을 선택하게 합니다.
6. 환경 API로 행동을 실행한 뒤 다시 관찰합니다.

이 순서는 [실제 실행기](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/scripts/planned-agent.mjs)에서 확인할 수 있습니다. 집 짓기·보관·장비 준비에는 별도 실행기가 있으므로, 이 파일 하나가 모든 플레이 과제를 처리한다고 읽으면 안 됩니다.

계획의 `count`는 “이번에 추가로 얻을 개수”가 아니라 해당 단계가 끝났을 때 필요한 최소 재고입니다. 예를 들어 판자 8개가 목표라면 실제 `oak_planks` 재고가 8개 이상인지 검사합니다. 재료가 이미 있으면 불필요한 수집 단계를 건너뛸 수 있고, 제작으로 앞 단계의 재료가 소비되는 것도 계획에서 고려해야 합니다. [계획 계약](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/planners/qwen.mjs), [완료 판정](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/planners/task-runner.mjs)

Qwen의 계획 갱신을 기다리는 동안에도 현재 수락한 계획으로 Kev의 행동 선택을 진행할 수 있습니다. 갱신된 답이 늦게 도착했는데 이미 다음 단계로 넘어갔다면, 실행기는 오래된 답을 버립니다. 계획의 표현을 갱신하더라도 수락한 아이템·개수·순서를 바꾸지 못하도록 검사합니다. 이 부분도 `TaskPlanner`에서 확인할 수 있습니다.

## 피해 대응은 모델 호출보다 먼저 시작합니다

Minecraft의 체력이 줄면 `installDamageReflex`는 이동 목표와 채굴을 중단하고 조작 상태를 해제합니다. 방패가 실제 보조 손 슬롯에 있다면 방어를 활성화합니다. **이 최초 중단은 Qwen이나 Kev의 응답을 기다리지 않는 로컬 코드입니다.** [damage-reflex.mjs](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/damage-reflex.mjs)

그다음 위협 대응 루프에서는 현재 체력, 장비, 주변 위협과 허용된 대응 행동을 Kev에게 보냅니다. 선택된 공격·후퇴·귀가 동작을 실행하고, 위협이 해소됐는지 다시 관찰합니다. 안전 대응이 해결되지 않거나 낮은 체력으로 대피한 상태라면 작업을 무조건 재개하지 않습니다. [task-safety.mjs](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/task-safety.mjs)

위협이 해소된 뒤에도 이전 행동을 모두 재생하지는 않습니다. 이동·귀가 등 재개 가능한 행동은 현재 후보를 다시 확인합니다. 제작이나 아이템 전송처럼 중복 실행의 영향이 있는 행동은 `action_needs_replan`으로 반환할 수 있습니다. 모든 실행기가 이 경우를 자동으로 다시 계획하는 것은 아닙니다. [행동 중단과 재개 구현](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/adapter.mjs)

이 구조 때문에 플레이를 ‘AI 모델 하나의 능력’으로만 설명하기 어렵습니다. 모델의 판단, 게임 상태를 읽는 코드, 실행 스킬, 즉시 중단, 재개 조건이 함께 결과를 만듭니다.

## 다른 과제에서 확인한 결과

야간 플레이 외에도 작은 집과 출입문·조명, 상자 보관, 철 장비 준비가 개발 실행에서 관찰됐습니다. 집에는 문과 횃불 네 개를 설치하고 내부 귀환과 출입구 폐쇄를 확인했습니다. 상자는 내용물을 다시 읽었고, 철 장비는 곡괭이·검·방패·갑옷 네 부위를 준비한 뒤 실제 장착 슬롯을 확인했습니다. [공개 결과표](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#observed-results-with-qwen38--kev-4b)

하지만 준비 과정에는 사망 후 장비 회수와 운영자의 낮 시간 조정·주변 적 제거가 포함됐습니다. 이것을 새 세계에서 시작한 무개입 생존이나 게임 클리어로 표현할 수는 없습니다. 이 프로젝트는 Ender Dragon 처치를 주장하지 않습니다.

## 직접 확인할 때의 실행 순서

다음은 공개 저장소가 제공하는 실행 절차입니다. 이 글에서는 명령을 다시 실행하지 않았습니다. 모델 가중치와 추론 서버는 별도 준비가 필요하며, Java 서버를 처음 만들 때는 Minecraft EULA를 직접 확인하고 동의 여부를 결정해야 합니다. [프로젝트 설치 안내](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#setup)

```bash
git clone https://github.com/LeeJeaHyuk/minecraft-agent-lab.git
cd minecraft-agent-lab
git checkout 7be99faac450f3031798a51185bd7d36ff79c958
npm ci
cp .env.example .env
node --env-file=.env scripts/prepare-server.mjs
```

EULA 확인 후 서버 준비를 마쳤다면, 별도 터미널에서 게임 서버와 환경 API를 각각 실행합니다.

```bash
# 게임 서버
npm run server

# 별도 터미널: Mineflayer 환경 API
npm start
```

`.env`의 `QWEN_BASE_URL`, `QWEN_MODEL`, `KEV_BASE_URL`, `KEV_REVISION`을 실제 추론 구성에 맞게 설정해야 합니다. Kev의 다운로드·시작 스크립트는 공개 README의 Models 절에 설명돼 있습니다. 샘플 설정은 모델 서버가 준비돼 있다는 뜻이 아닙니다.

게임과 두 모델이 준비된 뒤 아래 목표부터 확인할 수 있습니다.

```bash
npm run qwen:smoke
npm run agent -- kev 5
npm run planned -- "Collect two oak logs, then craft eight oak planks" 20
```

Minecraft Java 1.21.1 클라이언트로 `localhost:25565`에 접속하면 `LabBot`을 관찰할 수 있습니다. 공개 기본 설정은 loopback과 offline 인증을 사용하는 로컬 실험용입니다. 해당 설정을 인터넷에 그대로 노출해서는 안 됩니다. [실행 설정](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/.env.example), [서버 사용 범위](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#setup)

집·상자·장비·야간 대응을 보는 별도 실험은 앞선 준비 상태에 의존할 수 있습니다. 채집 예제를 실행했다고 집이나 장비 준비까지 자동으로 완료된다고 가정해서는 안 됩니다.

## 무엇을 기록해야 결과를 판단할 수 있을까요?

환경 API는 모델에게 준 상태와 후보, 선택된 행동, 확률, 다음 상태를 기록합니다. 계획 모델과 행동 선택 모델의 이름·revision·지연 시간도 구분합니다. 모델이 행동을 결정하는 데 걸린 시간과 실제 이동·제작에 걸린 시간을 나눠 봐야 병목을 찾을 수 있습니다. [기록 생성 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/api.mjs), [JSONL 기록 형식](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/logging/trajectory.mjs)

관찰용 게임 채팅에도 출처를 붙입니다. Qwen이 만든 공개 진행 발언, Kev의 선택에 대응한 메시지, 로컬 시스템의 중단 알림을 구분합니다. 이 발언을 모델의 비공개 추론 과정이라고 부르면 안 됩니다. OBS 녹화도 선택 기능이며, 다른 프로그램이 이미 시작한 녹화를 임의로 종료하지 않도록 소유권을 구분합니다. [채팅 출처 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/planners/chat-source.mjs), [녹화 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/recording/obs.mjs)

공개 README는 이 스냅샷이 자동 테스트 62개를 통과했다고 기록합니다. 테스트는 계약과 회귀를 확인하며, 플레이 성공률이나 일반적인 게임 능력의 증거와는 구분해야 합니다. 이번 글 작성에서 테스트와 플레이를 다시 실행한 결과로 보고하는 값은 아닙니다.

원본 로그와 녹화는 공개돼 있지 않습니다. 영상이나 로그를 이후 첨부할 때는 플레이어 이름, 채팅, 경로와 계정 정보 등을 먼저 확인해야 합니다. 이 초안은 공개된 코드와 결과 요약만 사용합니다. [프로젝트 공개 범위](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/PRIVACY.md)

## 다음 실행에서 확인할 부분

현재 기록으로 확인할 수 있는 것은 특정 개발 실행에서 채집·제작·전투·귀가를 연결했다는 점입니다. 크리퍼 회피, 식량 비축, 길찾기와 아이템 회수 확인은 여전히 개선이 필요한 부분으로 기록돼 있습니다.

반복 실험에서는 같은 목표와 고정된 모델 revision을 사용하고, 세계 조건·시작 재고·사망·운영자 개입을 함께 남겨야 합니다. 특히 ‘전투 뒤 귀가를 재개한 비율’과 ‘채집 타임아웃 후 실제 획득을 확인한 경우’를 따로 보면, 모델의 선택 문제와 실행기의 판정 문제를 구분하는 데 도움이 될 것으로 생각합니다. 이는 다음 검증을 위한 제안이며 현재 측정한 성공률은 아닙니다.
