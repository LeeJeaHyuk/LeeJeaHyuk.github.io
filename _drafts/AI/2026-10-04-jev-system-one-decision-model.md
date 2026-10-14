---
layout: single
post_type: concept
title: "Decision Model 개념 정리: Jev의 API와 Kev의 공개 구현"
date: 2026-10-04 00:00:00 +0900
last_modified_at: 2026-10-13
categories: [AI]
tags: [기술정리, 개념정리, Jev, Kev, DecisionModel, AIAgent, Minecraft]
permalink: /ai/jev-system-one-decision-model/
excerpt: "Jev의 choice·score·noul과 확률을 읽는 법, Kev의 공개 구현, 출력 제한과 판단 품질의 차이를 정리한다."
toc: true
toc_label: "이 글의 내용"
---

에이전트의 다음 행동을 프로그램에 연결하려면 출력 형식과 선택의 의미를 함께 정해야 한다. Jev는 상황을 읽고 미리 정의한 질문에 결정과 확률을 반환하는 TypeSafe의 모델이다. LLM·에이전트의 기초를 아는 개발자를 위해 질문 유형, 확률을 읽는 법, 공개 구현인 Kev와의 차이를 정리한다.

도입을 고민한 순서는 [결정 모델에 관심을 가진 생각 글](/ai/decision-model-introduction/)과 [게임으로 확장한 생각 글](/ai/minecraft-agent-lab-qwen-kev-gameplay/)에 담았다. 실제 실행은 [Minecraft 실험 기록](/ai/minecraft-agent-lab-experiment/), 모델과 외부 사례의 출처는 [자료 조사](/ai/decision-model-references/)에서 이어진다.

## 왜 다음 행동을 선택지로 받을까?

Minecraft 에이전트에게 “나무를 모아서 집을 지어라”라고 지시했다고 생각해 보자. 재료를 정하고 작업 순서를 만드는 단계에는 긴 계획이 필요하다. 하지만 이미 계획이 있는 상태에서 주변에 좀비가 나타났다면, 당장 필요한 출력은 짧다. 공격할지, 피할지, 집으로 돌아갈지를 선택하면 된다.

계획과 다음 행동 선택을 같은 주기로 호출하면 작은 판단에도 큰 계획을 반복해서 만들 수 있다. 선택지를 실행 가능한 행동으로 좁히고, 그중 하나를 고르는 작업을 별도로 두면 계획을 유지하면서 다음 동작을 바꿀 수 있다. 두 작업은 필요한 출력이 다르다는 점에서 구분할 수 있다.

TypeSafe가 공개한 System One 모델의 첫 제품이 Jev다. “System One”이라는 이름은 빠른 판단과 느린 숙고의 구분에서 가져왔다. [TypeSafe 공식 소개](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

## Jev가 제공하는 세 가지 질문

공식 API의 입력은 판단에 필요한 상황인 `state`와, 그 상황에서 물을 질문인 `questions`이다. 예를 들어 체력과 주변 위협을 `state`에 넣고, 다음 행동을 고르는 질문을 `questions`에 정의할 수 있다. 질문마다 필요한 답의 유형을 지정한다. [공식 Introduction](https://docs.typesafe.ai/introduction)

| 질문 유형 | 질문 예시 | 반환값을 읽는 방법 |
| --- | --- | --- |
| `choice` | “제시한 행동 중 무엇을 선택할까?” | `choice`와 선택지별 `probabilities`, `confidence`를 확인한다. |
| `score` | “이 상황의 위험 수준은 정의한 기준 중 어디인가?” | `score`, 기준을 설명하는 `legend`, 단계별 확률과 `confidence`를 확인한다. |
| `noul` | “관찰된 정보가 이 명제를 뒷받침하는가?” | `noul`은 ‘예’일 확률을 나타내는 0~1의 값이다. 별도의 `confidence`는 없다. |

`score`의 기준은 개발자가 정한다. 예를 들어 ‘안전함 / 주의 필요 / 즉시 대응’처럼 의미를 정의해야 숫자를 행동에 연결할 수 있다. `noul`이 0.5라는 것은 명제의 참·거짓이 불확실하다는 뜻이지, 대상의 수준이 중간이라는 뜻은 아니다. [공식 Primitives 문서](https://docs.typesafe.ai/primitives)

다음은 요청 예시이다. 집으로 돌아가기·방어하기·기다리기를 실행 가능한 행동 후보로 준비했다고 가정한다.

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

질문에 “안전하게 모든 일을 잘해 줘”라고 쓰는 것보다, 판단할 상황과 선택지를 구체적으로 주는 편이 프로그램에 연결하기 쉽다. 다만 좋은 행동을 후보에 넣는 일은 여전히 개발자의 책임이다. 집으로 돌아가는 행동을 누락했다면 모델은 그 행동을 선택할 수 없다.

## 같은 상황에 여러 질문을 묶을 수 있다

Jev는 같은 `state`에 대한 여러 질문을 한 요청으로 평가한다. 공식 문서에 따르면 각 질문은 같은 상태를 독립적으로 보고, 병렬로 처리된다. [공식 Introduction](https://docs.typesafe.ai/introduction)

Minecraft에 적용한다고 가정하면 ‘위협이 있는가’, ‘상태가 귀가를 요구하는가’, ‘현재 작업과 맞는 행동은 무엇인가’를 나눠 물을 수 있다. 코드가 결과를 조합하면 위험 대응의 우선순위도 명시적으로 정할 수 있다. 이를 구현한다면 각 질문의 결과를 조합하는 우선순위는 코드에서 정한다.

독립적인 질문이라는 점도 주의해야 한다. 같은 요청에 넣은 두 번째 질문이 첫 번째 질문의 답을 읽는 것은 아니다. 첫 답을 알아야 다음 선택지를 만들 수 있다면, 코드에서 그 결과를 받은 뒤 후속 요청을 해야 한다. [공식 Primitives 문서](https://docs.typesafe.ai/primitives)

## 확률과 confidence는 같은 숫자가 아니다

`probabilities`는 각 선택지에 부여한 확률이고, `confidence`는 그 분포에서 계산한 값이다. 둘의 차이를 보기 위해 다음 확률 분포를 가정해 보자.

```json
{
  "return_home": 0.7,
  "defend": 0.2,
  "wait": 0.1
}
```

가장 높은 확률은 0.7이다. 그러나 TypeSafe의 `choice`에 대한 `confidence`는 선택지 수를 고려해 균등 분포에서 얼마나 벗어났는지 계산한다. 선택지 수를 `n`, 가장 높은 확률을 `p_max`라고 하면 공식 문서의 식은 다음과 같다.

```text
confidence = (p_max - 1/n) / (1 - 1/n)
```

위 예시에서는 약 0.55다. 따라서 ‘최상위 선택지의 확률’과 API의 `confidence`를 서로 바꿔 쓰면 다른 기준으로 판단하게 된다. [공식 Confidence 문서](https://docs.typesafe.ai/confidence)

또한 모델이 한 선택지에 높은 확률을 줬다고 해서 게임에서 그 행동이 반드시 성공하지는 않는다. 이동 경로가 막혔거나, 판단 직후 몬스터가 접근했거나, 관찰에 필요한 정보가 빠졌을 수 있다. 확률에 따라 실행을 보류하거나 더 관찰하는 설계는 가능하지만, 그 기준은 해당 환경의 결과로 검증해야 한다.

## RLCD와 구조화된 답변을 어떻게 이해할까?

TypeSafe는 Jev를 RLCD, 즉 Reinforcement Learning for Calibrated Decisions로 학습했다고 설명한다. 판단과 확률을 출력하고, 높은 확률을 준 사건이 실제로 더 자주 맞도록 하는 것이 목표이다. 잘 보정된 모델이라면 많은 사례에서 0.8의 확률을 준 사건이 대략 80% 발생해야 한다. 이것은 개별 답변의 성공 보장이 아니라 여러 판단에 걸친 통계적 성질이다. [공식 AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)

일반 LLM도 구조화된 출력 기능으로 JSON과 선택지를 제한할 수 있다. 따라서 JSON을 반환한다는 사실만으로 새로운 모델의 효과가 입증되지는 않는다. 비교할 때는 같은 상태와 선택지를 주고, 판단 품질·전체 응답 시간·보정 정도를 따로 봐야 한다.

TypeSafe가 강조하는 형식 보장은 선택지와 출력 스키마를 벗어나지 않는다는 의미로 읽어야 한다. 후보 중 잘못된 행동을 고르는 문제까지 사라진다고 해석할 수는 없다. 공식 발표에도 스키마 보장과 평가 조건에 대한 설명이 함께 있다. [공식 소개의 형식 보장 설명](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

## Jev와 공개 결정 모델의 관계

Kev는 Jared Palmer가 공개한 별도의 Jev 유사 결정 모델 계열이다. Qwen 기반 모델에 결정 출력용 학습과 구현을 더하고, TypeSafe의 System One API 형식에 맞춰 실행할 수 있게 만들었다. Jev와 이름이나 인터페이스가 비슷해도 같은 가중치나 학습법을 공유한다는 뜻은 아니다. [Kev 프로젝트](https://github.com/jaredpalmer/kev)

Kev-4B는 Qwen3.5-4B-Base에 LoRA와 pointer head를 결합한다. pointer head는 주어진 선택지에 점수를 매기는 출력부이다. 문장을 생성하는 대신 선택지별 점수에서 확률을 계산하며, 같은 상태에 대한 질문들은 상태 계산을 공유한다. 이 구조는 공개된 Kev의 구현 설명이며 Jev의 내부 구조를 단정하는 근거로 쓰지는 않는다. [Kev-4B 모델 카드](https://huggingface.co/jaredpalmer/kev-4b)

로컬 실행 비용은 모델 크기와 입력 길이, 정밀도에 따라 달라진다. Kev-0.8B 카드에는 4GB GPU 실행이 안내돼 있지만, Kev-4B 카드는 24GB GPU 또는 32GB Apple Silicon Mac을 예로 든다. 따라서 ‘결정 모델은 4GB면 된다’고 묶어 말할 수 없다. [Kev-0.8B 모델 카드](https://huggingface.co/jaredpalmer/kev-0.8b), [Kev-4B 모델 카드](https://huggingface.co/jaredpalmer/kev-4b)

## 선택 범위를 제한하는 것과 결정론은 다르다

여기서 결정 모델이라는 이름은 결정론적 알고리즘을 뜻하지 않는다. 허용한 선택지를 벗어나지 않는다는 성질과, 같은 요청에서 언제나 완전히 같은 확률과 답을 준다는 성질은 별개다. 독립적인 Jev API 조사에서도 반복 요청의 작은 차이와 선택지 순서의 영향을 다뤘다. [Archer Hume의 API 조사](https://archerhume.com/posts/jevs-architecture-unmasked)

실행을 설계할 때는 출력이 후보에 속하는지 확인하고, 선택이 틀린 경우와 행동 실행이 실패한 경우를 구분해야 한다. 관찰에 필요한 정보가 빠졌거나 좋은 행동이 후보에서 누락된 문제는 출력 형식만으로 해결되지 않는다. 작은 판단을 분리하려면 상태·후보·실행 결과를 함께 기록하고, 같은 조건에서 기존 방식과 비교하는 것이 출발점이다.
