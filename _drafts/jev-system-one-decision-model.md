---
layout: single
title: "Jev의 특성 이해하기: 선택과 확률을 반환하는 결정 모델"
date: 2026-10-04 00:00:00 +0900
categories: [AI]
tags: [Jev, Kev, DecisionModel, AIAgent, Minecraft]
permalink: /ai/jev-system-one-decision-model/
excerpt: "Jev의 Choice·Score·Noul, 확률과 confidence의 차이, 그리고 Minecraft Agent Lab에서 Kev를 사용하는 이유를 살펴봅니다."
toc: true
toc_label: "이 글의 내용"
draft: true
---

Jev를 이해할 때 먼저 볼 것은 답변의 모양입니다. Jev는 상황과 질문을 받아, 미리 정한 선택지나 평가 기준에 대한 결정과 확률을 반환합니다. 프로그램은 그 값을 받아 다음 동작을 정할 수 있습니다. TypeSafe가 2026년 9월 15일 공개한 System One 모델의 첫 제품이 Jev입니다. [TypeSafe 공식 소개](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

이 글에서는 이 방식이 에이전트에 어떤 의미가 있는지 살펴보겠습니다. 구체적인 연결 사례는 [Minecraft Agent Lab](https://github.com/LeeJeaHyuk/minecraft-agent-lab/tree/7be99faac450f3031798a51185bd7d36ff79c958)입니다. **이 저장소의 플레이 모델은 Jev 자체가 아니라 Qwen3.8-27B와 Kev-4B입니다.** Jev의 특성과 Jev와 유사한 결정 인터페이스를 제공하는 Kev의 실제 사용을 구분해 읽어야 합니다.

> 초안입니다. 공식 문서와 프로젝트의 공개 소스를 2026-10-04 기준으로 확인했습니다. 이 글을 작성하면서 Jev API의 속도나 정확도를 직접 측정하지는 않았습니다.

## 에이전트가 매번 문장을 만들 필요가 있을까요?

Minecraft 에이전트에게 “나무를 모아서 집을 지어라”라고 지시했다고 생각해 보겠습니다. 재료를 정하고 작업 순서를 만드는 단계에는 긴 계획이 필요합니다. 하지만 이미 계획이 있는 상태에서 주변에 좀비가 나타났다면, 당장 필요한 출력은 짧습니다. 공격할지, 피할지, 집으로 돌아갈지를 선택하면 됩니다.

이때 계획과 다음 행동 선택을 같은 주기로 호출하면, 작은 판단에도 큰 계획을 반복해서 만들 수 있습니다. 선택지를 실행 가능한 행동으로 좁히고, 그중 하나를 고르는 작업을 별도로 두면 계획을 유지하면서 다음 동작을 바꿀 수 있습니다. 이것이 Minecraft Agent Lab을 읽으며 주목한 설계입니다.

| 맡길 일 | 필요한 출력 | 이 프로젝트의 담당 |
| --- | --- | --- |
| 목표를 순서 있는 하위 작업으로 만들기 | 목표, 재료, 개수, 작업 순서 | QwenPlanner |
| 현재 하위 작업에서 다음 행동 고르기 | 허용된 행동 ID와 확률 분포 | KevPolicy |
| 이동·채집·제작·전투를 실행하기 | 게임 상태의 변화와 실행 결과 | Mineflayer 기반 환경 코드 |
| 피해를 받았을 때 진행 중인 동작 중단하기 | 즉시 중단과 방어 동작 | 로컬 damage reflex |

이 역할 구분은 [계획 실행 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/scripts/planned-agent.mjs)와 [피해 반응 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/damage-reflex.mjs)에서 확인할 수 있습니다. “System One”이라는 이름도 빠른 판단과 느린 숙고의 구분에서 가져왔지만, 이 표는 인간의 사고를 재현했다는 주장이 아니라 실제 프로그램의 역할 분담을 설명합니다.

## Jev가 제공하는 세 가지 질문

공식 API는 `state`와 `questions`를 받습니다. `state`에는 판단에 필요한 상황을 넣고, 각 질문에는 어떤 종류의 답이 필요한지 지정합니다. [공식 Introduction](https://docs.typesafe.ai/introduction)

| 질문 유형 | 질문 예시 | 반환값을 읽는 방법 |
| --- | --- | --- |
| `choice` | “제시한 행동 중 무엇을 선택할까요?” | `choice`와 선택지별 `probabilities`, `confidence`를 확인합니다. |
| `score` | “이 상황의 위험 수준은 정의한 기준 중 어디인가요?” | `score`, 기준을 설명하는 `legend`, 단계별 확률과 `confidence`를 확인합니다. |
| `noul` | “관찰된 정보가 이 명제를 뒷받침하나요?” | `noul`은 ‘예’일 확률을 나타내는 0~1의 값입니다. 별도의 `confidence`는 없습니다. |

`score`의 기준은 개발자가 정합니다. 예를 들어 ‘안전함 / 주의 필요 / 즉시 대응’처럼 의미를 정의해야 숫자를 행동에 연결할 수 있습니다. `noul`이 0.5라는 것은 명제의 참·거짓이 불확실하다는 뜻이지, 대상의 수준이 중간이라는 뜻은 아닙니다. [공식 Primitives 문서](https://docs.typesafe.ai/primitives)

아래는 동작을 설명하기 위해 만든 요청 예시입니다. **실제 플레이 로그나 Minecraft Agent Lab의 실제 행동 ID를 옮긴 것이 아닙니다.** 실행 가능한 행동인지 먼저 확인했다는 가정 아래에서 요청 모양만 보여줍니다.

```json
{
  "state": {
    "health": 7,
    "nearby_threat": "zombie",
    "shelter_available": true,
    "goal": "collect wood"
  },
  "questions": {
    "next_action": {
      "type": "choice",
      "instructions": "Choose the next action from the available options.",
      "criteria": {
        "return_home": "Move to the available shelter.",
        "defend": "Defend against the nearby zombie.",
        "wait": "Wait briefly at the current position."
      }
    }
  }
}
```

질문에 “안전하게 모든 일을 잘해 주세요”라고 쓰는 것보다, 판단할 상황과 선택지를 구체적으로 주는 편이 프로그램에 연결하기 쉽습니다. 다만 좋은 행동을 후보에 넣는 일은 여전히 개발자의 책임입니다. 집으로 돌아가는 행동을 누락했다면 모델은 그 행동을 선택할 수 없습니다.

## 같은 상황에 여러 질문을 묶을 수 있습니다

Jev는 같은 `state`에 대한 여러 질문을 한 요청으로 평가합니다. 공식 문서에 따르면 각 질문은 같은 상태를 독립적으로 보고, 병렬로 처리됩니다. [공식 Introduction](https://docs.typesafe.ai/introduction)

Minecraft에 적용한다고 가정하면 ‘위협이 있는가’, ‘상태가 귀가를 요구하는가’, ‘현재 작업과 맞는 행동은 무엇인가’를 나눠 물을 수 있습니다. 코드가 결과를 조합하면 위험 대응의 우선순위도 명시적으로 정할 수 있습니다. 이는 적용 아이디어이며, 현재 Lab이 이 세 질문을 한꺼번에 보내고 있다는 뜻은 아닙니다.

독립적인 질문이라는 점도 주의해야 합니다. 같은 요청에 넣은 두 번째 질문이 첫 번째 질문의 답을 읽는 것은 아닙니다. 첫 답을 알아야 다음 선택지를 만들 수 있다면, 코드에서 그 결과를 받은 뒤 후속 요청을 해야 합니다. [공식 Primitives 문서](https://docs.typesafe.ai/primitives)

## 확률과 confidence는 같은 숫자가 아닙니다

설명용으로 다음 확률 분포를 가정하겠습니다.

```json
{
  "return_home": 0.7,
  "defend": 0.2,
  "wait": 0.1
}
```

가장 높은 확률은 0.7입니다. 그러나 TypeSafe의 `choice`에 대한 `confidence`는 선택지 수를 고려해 균등 분포에서 얼마나 벗어났는지 계산합니다. 선택지 수를 `n`, 가장 높은 확률을 `p_max`라고 하면 공식 문서의 식은 다음과 같습니다.

```text
confidence = (p_max - 1/n) / (1 - 1/n)
```

위 예시에서는 약 0.55입니다. 따라서 ‘최상위 선택지의 확률’과 API의 `confidence`를 서로 바꿔 쓰면 다른 기준으로 판단하게 됩니다. [공식 Confidence 문서](https://docs.typesafe.ai/confidence)

또한 모델이 한 선택지에 높은 확률을 줬다고 해서 게임에서 그 행동이 반드시 성공하지는 않습니다. 이동 경로가 막혔거나, 판단 직후 몬스터가 접근했거나, 관찰에 필요한 정보가 빠졌을 수 있습니다. 확률에 따라 실행을 보류하거나 더 관찰하는 설계는 가능하지만, 그 기준은 해당 환경의 결과로 검증해야 합니다.

현재 Lab의 `KevPolicy`는 선택된 행동 ID, 확률의 범위, 확률 합계를 검사합니다. Jev 문서의 `confidence`에 따른 자동 승격·보류 정책을 구현한 것으로 읽으면 안 됩니다. 해당 코드가 주로 사용하는 것은 `choice`와 `probabilities`입니다. [정책 코드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/policies/index.mjs)

## RLCD와 구조화된 답변을 어떻게 이해할까요?

TypeSafe는 Jev를 RLCD, 즉 Reinforcement Learning for Calibrated Decisions로 학습했다고 설명합니다. 판단과 확률을 출력하고, 높은 확률을 준 사건이 실제로 더 자주 맞도록 하는 것이 목표입니다. 잘 보정된 모델이라면 많은 사례에서 0.8의 확률을 준 사건이 대략 80% 발생해야 합니다. 이것은 개별 답변의 성공 보장이 아니라 여러 판단에 걸친 통계적 성질입니다. [공식 AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)

일반 LLM도 구조화된 출력 기능으로 JSON과 선택지를 제한할 수 있습니다. 따라서 JSON을 반환한다는 사실만으로 새로운 모델의 효과가 입증되지는 않습니다. 비교할 때는 같은 상태와 선택지를 주고, 판단 품질·전체 응답 시간·보정 정도를 따로 봐야 합니다.

TypeSafe가 강조하는 형식 보장은 선택지와 출력 스키마를 벗어나지 않는다는 의미로 읽어야 합니다. 후보 중 잘못된 행동을 고르는 문제까지 사라진다고 해석할 수는 없습니다. 공식 발표에도 스키마 보장과 평가 조건에 대한 설명이 함께 있습니다. [공식 소개의 형식 보장 설명](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

## Jev와 Kev를 구분해야 하는 이유

Jev는 TypeSafe의 모델입니다. Kev는 Jared Palmer가 공개한 별도의 Jev 유사 결정 모델 계열이며, 직접 실행하거나 학습할 수 있도록 공개한 구현입니다. 두 이름이 비슷하고 API 형태가 닮았더라도 같은 가중치나 같은 성능을 가진 모델이라는 뜻은 아닙니다. [Kev 프로젝트](https://github.com/jaredpalmer/kev)

Minecraft Agent Lab의 공개 기록에서는 다음 모델을 사용했습니다.

| 모델 | 역할 | 기록된 실행 구성 |
| --- | --- | --- |
| Qwen3.8-27B | 장기 목표와 하위 작업 계획 | OpenAI-compatible endpoint, GGUF 변형, 요청 수준 thinking 비활성화 |
| jaredpalmer/kev-4b | 제한된 행동 후보 중 선택 | 공식 adapter/head, BF16, 추가 양자화 없음 |
| Qwen/Qwen3.5-4B-Base | Kev의 기반 모델 | Kev 구성에 포함되며 별도 계획 모델은 아님 |

이 구성과 고정된 모델 revision은 [Lab의 모델 기록](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/README.md#models)에 명시돼 있습니다. Jev의 공개 자료를 읽으며 관심을 가진 ‘상황을 보고 제한된 결정을 반환한다’는 방식을, Lab에서는 Kev와 Qwen의 역할 분리로 살펴볼 수 있습니다.

## Minecraft 사례에서 확인할 것

이 설계를 평가할 때는 행동 선택 모델만 보지 않으려고 합니다. 유효한 후보를 만들었는지, 실제 실행 전에 상태를 다시 확인했는지, 피해를 받으면 긴 작업을 중단하는지, 작업 완료를 재고와 장착 상태로 검증하는지가 함께 중요합니다.

속도 역시 모델 추론 시간과 게임 동작 시간을 분리해야 합니다. 빠르게 채집을 선택해도 나무까지 이동하고 아이템을 줍는 데는 시간이 걸립니다. Lab의 기록 코드도 결정 지연과 행동 실행 시간을 구분해 저장합니다. [환경 API와 기록 필드](https://github.com/LeeJeaHyuk/minecraft-agent-lab/blob/7be99faac450f3031798a51185bd7d36ff79c958/environment/api.mjs)

다음 글에서는 이 구조가 야간 플레이에서 어떻게 작동했고, 어디에서 부족함이 드러났는지 살펴보겠습니다. [Qwen과 Kev로 Minecraft를 플레이하기](/ai/minecraft-agent-lab-qwen-kev-gameplay/)
