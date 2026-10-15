---
layout: single
post_type: reference
title: "로컬 모델 및 결정 모델 적용 사례"
date: 2026-10-12 00:00:00 +0900
last_modified_at: 2026-10-15
categories: [AI]
tags: [자료조사, LocalLLM, Jev, Kev, OpenJev, Mica, DecisionModel, AIAgent, Minecraft, Skyrim, Fallout4, NPC]
permalink: /ai/decision-model-references/
excerpt: "로컬 모델과 결정 모델의 업무·게임 적용 사례를 수집일, 원문과 실행 조건을 붙여 누적한다."
toc: true
toc_label: "이 글의 내용"
---

공개 모델과 적용 사례를 수집일과 함께 누적한다. 수집일은 자료를 확인한 날짜이며, 사례가 처음 공개된 날짜와는 구분한다.

관심을 가진 계기와 적용 과정은 [Decision Model - Jev을 발견, 조사, 테스트 (1)](/ai/decision-model-introduction/)에 담았다.

## 연결 문서

- [Jev·Kev 개념 정리](/ai/jev-system-one-decision-model/)
- [판단 작업 교체 실험 기록](/ai/decision-model-routing-experiment/)
- [Minecraft 에이전트 실험 기록](/ai/minecraft-agent-lab-experiment/)

<a id="cases"></a>

## 수집한 적용 사례

| 수집일 | 사례 | 맡긴 역할과 보고된 내용 | 조건·원문 |
| --- | --- | --- | --- |
| 2026-10-15 | Mantella | Skyrim·Fallout 4의 NPC 대화에 음성 인식, LLM, 음성 합성을 연결한다. | 공식 안내는 로컬 모델과 외부 API를 모두 지원한다. [공식 소개](https://art-from-the-machine.github.io/Mantella/), [코드](https://github.com/art-from-the-machine/Mantella/tree/c0c1ae6db01f3c2a14dbab8ec5c3e6419e4f6bac) |
| 2026-10-15 | CHIM | Skyrim의 NPC 대화·세계 상태·게임 내 행동을 연결한다. | 공식 가이드는 DwemerDistro 내부의 LLM Studio 또는 별도 LM Studio·KoboldCPP 연결을 안내한다. [로컬 모델 가이드](https://github.com/Dwemer-Dynamics/DwemerDynamics/blob/89aa67ae371e2e33d27cd576921dc1d5fdfd3820/chim/local-llm-guide.html), [기능 소개](https://github.com/Dwemer-Dynamics/DwemerDynamics/blob/89aa67ae371e2e33d27cd576921dc1d5fdfd3820/chim/index.html) |
| 2026-10-15 | [Astra・Jev Minecraft agent](https://github.com/rmalde/minecraft-agent/tree/78b40ed59514e5e2abde33a05ce398ecb2c39e05) | Astra가 목표를 계획하고 Jev가 행동을 선택한다. 제작자는 Ender Dragon 처치 실행을 기록한다. | Survival·Peaceful 설정과 정해진 seed·알려진 좌표를 사용한다. Mineflayer가 이동과 게임 프로토콜을 처리한다. 녹화와 원본 실행 증거는 로컬에 보관하며 저장소에는 포함하지 않는다. |
| 2026-10-12 | [Kev 고객 문의 추가 학습](https://github.com/jaredpalmer/kev/blob/2ecafc49fe1e8945cd600443e6c04c1cf0b3a78f/skills/kev-finetune/README.md) | 문의에 대한 세 질문을 학습한 예제에서 정확도 67.7% → 73.6%, 오류 예산 5%일 때 자동 처리 비중 34% → 48%를 보고했다. | 생성 데이터 1,050개, H100에서 15분 학습한 특정 업무 예제이다. 데이터 400개에서는 개선이 잡음 범위 안이었다. |
| 2026-10-12 | [Kev 금융 민원 데이터 학습](https://huggingface.co/jaredpalmer/kev-4b) | CFPB 민원 문서로 상품·주요 문제를 분류하는 질문을 학습했다. | 학습 기록의 실제 민원 5,219개는 투자 예측이나 수익률 실험을 뜻하지 않는다. 다른 문서로 옮겼을 때의 품질은 별도로 봐야 한다. |
| 2026-10-12 | [OpenJev 브라우저 단계 평가](https://huggingface.co/alanhuangya/OpenJev-1.7B) | 다음 조작과 대상 요소를 고르는 데 사용했다. Mind2Web 평가에서 두 선택을 모두 맞힌 단계 비율 68.2%를 보고했다. | 873개 요청의 단계 평가이다. 후보 요소는 에이전트가 추렸고, 이는 브라우저 과제 전체의 완료율이 아니다. Jev 자체와는 비교하지 않았다. |
| 2026-10-12 | [Mica Tetris 데모](https://huggingface.co/sky7350/Mica-v0.1-4B#demo-tetris-mica-vs-n) | 같은 조각과 선택지로 진행한 쉬운 설정에서 세 seed의 제거 줄 합계가 Mica 223, Kev-4B 55, Laya 17이었다. | 세 seed의 특정 게임 설정이다. 코드·로그·영상 생성기가 공개돼 있지만 다른 게임이나 NPC의 품질로 일반화할 수 없다. |
| 2026-10-12 | [TypeSafe Doom·Wikiracing 데모](https://typesafe.ai/blog/introducing-system-one-models-and-jev#fun-demos) | 게임 행동 또는 현재 페이지의 링크를 골랐다. | Doom은 구조화된 상태를 받으며 화면 인식 데모가 아니다. Wikiracing은 많은 후보에서 점수화 후 선택하는 두 단계도 사용했다. |
| 2026-10-12 | [jevman Pac-Man 비교](https://github.com/opper-ai/jevman-benchmark/tree/b202e2cefd3dcf62b99835b319450b687960f40a) | 이동 선택과 위험 회피를 결정 모델에 맡기고 게임 기록·지연·점수를 비교한다. | 응답이 제때 오지 않으면 규칙 기반 동작이 대신한다. 모델 점수와 함께 이런 대체 동작의 조건도 읽어야 한다. |

<!-- 새 사례는 수집일·역할·보고된 결과·조건·원문 버전을 붙여 이 표에 추가한다. 같은 사례를 다시 확인할 때도 이전 수집일과 기록을 보존한다. -->

### Mantella 추가 학습 자료 — 2026-10-15 수집

Mantella는 Skyrim의 플레이어·NPC 대화 데이터와 Llama 3 8B 추가 학습 모델을 공개했다. 현재 공식 문서는 이 모델이 오래되어 최신의 추가 학습하지 않은 모델보다 성능이 낮으며 사용을 권장하지 않는다고 명시한다. [추가 학습 안내](https://art-from-the-machine.github.io/Mantella/pages/fine_tuned_models.html)

모드의 로컬 실행 지원, 게임 전용 추가 학습 모델의 존재, 현재 권장 모델은 각각 확인한다. Mantella와 CHIM이 같은 모델을 쓰거나 같은 판단 방식을 사용한다는 의미는 아니다.

## 공개 모델과 제공 범위 — 2026-10-12 수집

| 모델·구현 | 확인한 제공 형태 | 다시 볼 조건과 원문 |
| --- | --- | --- |
| Jev / TypeSafe | 호스팅된 System One API | 공개 Kev 가중치와 별개다. ‘환각 없음’의 근거는 출력 스키마 보장이다. [공식 소개](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [API](https://docs.typesafe.ai/introduction) |
| Jared Palmer의 Kev | Qwen 기반 구현과 0.8B·4B·9B·27B 모델 | 자기 라벨로 추가 학습할 수 있다. 조사한 저장소 버전은 `2ecafc49`이다. [고정 버전](https://github.com/jaredpalmer/kev/tree/2ecafc49fe1e8945cd600443e6c04c1cf0b3a78f), [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) |
| OpenJev-1.7B | 로컬 결정 API, 코드·가중치·학습 및 평가 설명 | TypeSafe와 독립된 프로젝트이며 브라우저 조작·대상 선택도 다룬다. [모델 카드](https://huggingface.co/alanhuangya/OpenJev-1.7B), [코드](https://github.com/alanhuangyoo/OpenJev) |
| Mica v0.1 4B | Qwen3.5-4B 기반 BF16·GGUF 가중치와 서버 코드 | 모델 카드가 지정한 가중치는 `ca36594`이다. 언어·입력 길이·실행 정밀도에 따라 평가를 나눠 읽는다. [모델 카드](https://huggingface.co/sky7350/Mica-v0.1-4B), [코드](https://github.com/akivet/Mica-v0.1-4B) |

작은 모델이라고 모두 같은 GPU에서 실행되지는 않는다. [Kev-0.8B 카드](https://huggingface.co/jaredpalmer/kev-0.8b)의 4GB GPU 안내와 [Kev-4B 카드](https://huggingface.co/jaredpalmer/kev-4b)의 24GB GPU 예시는 다른 구성을 가리킨다. 기반 모델·adapter·출력부·캐시와 게임 실행에 필요한 메모리를 함께 봐야 한다.

## API와 아키텍처를 읽을 때 필요한 자료

질문 유형은 [Primitives](https://docs.typesafe.ai/primitives), 확률에서 `confidence`를 계산하는 방법은 [Confidence](https://docs.typesafe.ai/confidence), RLCD와 보정의 취지는 [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)에 정리돼 있다.

[Archer Hume의 조사](https://archerhume.com/posts/jevs-architecture-unmasked)는 Jev API를 관찰해 내부 구조를 추론하고 반복 요청·선택지 순서의 영향을 다룬다. API에서 관찰한 현상과 제안한 아키텍처 가설을 구분해 읽는다. Kev의 공개 구조가 Jev의 비공개 구현과 같다는 근거로 옮겨 쓰지 않는다.

## 검색과 확인 기록

### 2026-10-12 검색 질문

| 찾으려던 내용 | 사용한 검색어 | 읽은 자료 |
| --- | --- | --- |
| Jev와 공개 구현은 어떤 관계인가? | `Jev Kev decision model open source Hugging Face` | TypeSafe 소개·API 문서, Kev 저장소와 모델 카드, OpenJev·Mica 모델 카드 |
| 자기 데이터로 학습해 개선한 사례가 있는가? | `site:github.com/jaredpalmer/kev finetune accuracy CFPB` | Kev의 추가 학습 예제와 모델 카드 |
| ‘결정 모델’은 같은 답을 보장하는 결정론적 구조인가? | `Jev architecture deterministic Archer Hume` | Archer Hume의 Jev API 조사 |

검색 결과에서 원문을 따라 읽었고, 게임 사례는 공식 소개와 연결된 저장소까지 확인했다.

### 추가 확인 경로와 버전

| 수집일 | 검색어·확인 경로 | 확인 자료·버전 |
| --- | --- | --- |
| 2026-10-15 | `Mantella Skyrim Fallout 4 local LLM GitHub` | Mantella 코드 `c0c1ae6d`, 공식 소개와 Fine-Tuned Models 문서. 문서 브랜치 버전은 [71e93802](https://github.com/art-from-the-machine/Mantella/tree/71e93802abbeaf1b8a60d20ef7a3b749e8c6ae63)이다. |
| 2026-10-15 | `CHIM Skyrim local LLM GitHub` | DwemerDynamics 문서 `89aa67ae`의 CHIM 소개와 Local LLM Guide. |
| 2026-10-12 | 기존 자료 조사에서 게임 행동 선택 사례를 분류했다. | Mica 모델 카드, TypeSafe 공식 데모, jevman `b202e2ce`. |
| 2026-10-15 | 생각 글에 적힌 `https://github.com/rmalde/minecraft-agent` 원문을 확인했다. | Minecraft agent `78b40ed5`의 README. |

## 내 프로젝트의 기록과 연결

Minecraft Agent Lab은 [7be99fa 버전](https://github.com/LeeJeaHyuk/minecraft-agent-lab/tree/7be99faac450f3031798a51185bd7d36ff79c958)의 Qwen3.8-27B·Kev-4B 조합을 다룬다. 실행 조건과 야간 채집·전투·귀가 결과, 크리퍼 피해와 운영자 개입은 [Minecraft 실험 기록](/ai/minecraft-agent-lab-experiment/)에 모았다.

## 후속 조사에서 확인할 것

- 모델을 바꿀 때 가중치 revision, 기반 모델, 언어와 컨텍스트 조건을 확인한다.
- 자기 업무의 고정된 평가 데이터에서 기존 방식과 정확도·보정·전체 작업 시간을 비교한 자료를 찾는다.
- 게임 사례는 판단 시간과 행동 실행 시간을 나누고, 규칙 기반 대체 동작과 운영자 개입을 확인한다.
- 더 작은 모델이나 양자화를 쓴 결과는 같은 입력에서 품질과 메모리 사용을 함께 비교한다.
- Jev의 첫 공개일은 날짜가 명시된 공식 자료가 있는지 다시 확인한다. 현재 소개 페이지에는 공개일이 표시되지 않는다.

## 수집 이력

| 수집일 | 추가·수정한 내용 |
| --- | --- |
| 2026-10-12 | Jev·Kev·OpenJev·Mica의 공개 자료, 업무 적용과 Tetris·Doom·Pac-Man 사례를 수집했다. |
| 2026-10-15 | Mantella·CHIM과 Astra·Jev Minecraft 사례를 추가하고, 사례별 수집일을 표시한 누적 목록으로 통합했다. |
